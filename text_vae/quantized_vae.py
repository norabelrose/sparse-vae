from .core import LanguageModel, LanguageModelHparams, Quantizer
from .funnel_transformer import FunnelTransformer, FunnelTransformerHparams
from .text_data_module import *
from .train_callbacks import ReconstructionSampler
from dataclasses import dataclass, field

from copy import deepcopy
from functools import partial
from numpy import prod
from omegaconf import DictConfig
from torchtext.data.metrics import bleu_score
from torch import nn, Tensor
from typing import *
import torch


@dataclass
class QuantizedVAEHparams(LanguageModelHparams):
    funnel: FunnelTransformerHparams = FunnelTransformerHparams(
        d_model=512,
        num_heads=8,
        block_sizes=(6,),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=tuple(),  # How much the hidden state is downsampled between each encoder block
    )

    beta: float = 0.25
    codebook_size: int = 1024
    ema_decay: float = 0.99

    autoregressive: bool = False
    include_full_res_latents: bool = True

    latent_depth: int = 64
    lr: float = 3e-4  # From the OpenAI Jukebox VQ-VAE hyperparameters


@dataclass
class QuantizedVAEState:
    code_indices: List[Tensor] = field(default_factory=list)
    soft_codes: List[Tensor] = field(default_factory=list)

    decoder_input: Optional[Tensor] = None
    encoder_states: List[PaddedTensor] = field(default_factory=list)

    commitment_loss: Union[float, Tensor] = 0.0
    embedding_loss: Union[float, Tensor] = 0.0
    logits: Optional[Tensor] = None


class QuantizedVAE(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(QuantizedVAE, self).__init__(hparams)

        funnel_hparams = hparams.funnel
        decoder_hparams = deepcopy(funnel_hparams)
        decoder_hparams.update(
            block_sizes=list(reversed(funnel_hparams.block_sizes)),
            scaling_factors=list(reversed(funnel_hparams.scaling_factors)),
            upsampling=True
        )
        self.encoder = FunnelTransformer(funnel_hparams)
        self.decoder = FunnelTransformer(decoder_hparams)

        d_model = funnel_hparams.d_model
        output_embedding = nn.Linear(d_model, hparams.vocab_size)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            output_embedding
        )
        self.encoder.input_layer[0].weight = output_embedding.weight

        num_latent_scales = len(hparams.funnel.scaling_factors)
        if hparams.include_full_res_latents:
            num_latent_scales += 1

        # These combine the encoder and decoder states together right before quantization
        d_model = hparams.funnel.d_model
        self.combiners = nn.ModuleList([
            nn.Linear(d_model * 2, d_model)
            for _ in range(num_latent_scales - 1)
        ])
        self.quantizers = nn.ModuleList([
            Quantizer(
                num_codes=hparams.codebook_size,
                code_depth=hparams.latent_depth,
                d_model=d_model,
                ema_decay=hparams.ema_decay
            )
            for _ in range(num_latent_scales)
        ])
        self.gathering_latents = False

        blocks = self.decoder.hparams.block_sizes[1:1 + num_latent_scales]
        if blocks:
            self.decoder.configure_cross_attention([(block, 0) for block, length in enumerate(blocks, start=1)])

        self.initialize_weights()

    def setup(self, stage: str):
        super().setup(stage)

        data: TextDataModule = self.trainer.datamodule if stage == 'fit' else self.datamodule
        data.hparams.pad_to_multiple_of = int(prod(self.decoder.hparams.scaling_factors))

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return callbacks + [ReconstructionSampler()]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        pbar_dict = super().get_progress_bar_dict()
        del pbar_dict['loss']  # We don't want to include the commitment and embedding losses in the progress bar
        return pbar_dict

    def configure_optimizers(self, **kwargs):
        codebooks = set(x.codebook for x in self.quantizers)
        noncodebook_params = list(filter(lambda x: x not in codebooks, self.parameters()))

        # Use a higher learning rate for the codebook so that 'dead' codes move quickly toward the
        # encoder outputs and get revived
        return super().configure_optimizers(params=[
            {'params': list(codebooks), 'lr': 10.0 * self.hparams.lr},
            {'params': noncodebook_params}
        ])

    def on_train_epoch_start(self):
        for quantizer in self.quantizers:
            quantizer.reset_code_usage_info()

    # quantize = False is used by update_codebook_kmeans()
    def forward(self, batch: Dict[str, PaddedTensor]) -> QuantizedVAEState:
        encoder_hiddens = self.encoder(batch['token_ids'])
        quant_out = self.quantizers[0](encoder_hiddens[-1])

        vae_state = QuantizedVAEState()
        vae_state.commitment_loss = quant_out.commitment_loss
        vae_state.embedding_loss = quant_out.embedding_loss

        encoder_states = encoder_hiddens[:-1]
        encoder_states.reverse()
        if not self.hparams.include_full_res_latents:
            del encoder_states[-1]

        vae_state.decoder_input = self.quantizers[0].upsample_codes(quant_out.hard_codes)
        vae_state.encoder_states = encoder_states
        vae_state.code_indices = [quant_out.code_indices]
        vae_state.soft_codes = [quant_out.soft_codes]

        callback = partial(self.decoder_block_end, vae_state)
        decoder_hiddens = self.decoder(vae_state.decoder_input, block_end_callback=callback)
        vae_state.logits = self.output_layer(decoder_hiddens[-1]) if decoder_hiddens else None

        return vae_state

    def decoder_block_end(self, vae_state: QuantizedVAEState, block_idx: int, dec_state: Tensor):
        cross_attn_kv = vae_state.encoder_states[block_idx - 1] if block_idx > 0 else None
        if block_idx >= len(vae_state.encoder_states):
            return dec_state, cross_attn_kv

        enc_state = vae_state.encoder_states[block_idx]

        # It might be better to use cross attention here
        x = self.combiners[block_idx](torch.cat([dec_state, enc_state], dim=-1))
        quantizer_out = self.quantizers[block_idx + 1](x)
        vae_state.code_indices.append(quantizer_out.code_indices)

        # End the forward pass early if we're gathering latent codes and we just computed the bottom latent feature map
        if self.gathering_latents and block_idx + 2 == len(self.quantizers):
            return None

        # Don't use += operator to avoid autograd errors from doing an in-place operation- weirdly
        # this only seems to happen when we train with full precision
        old_commit_loss, old_embed_loss = vae_state.commitment_loss, vae_state.embedding_loss
        vae_state.commitment_loss = old_commit_loss + quantizer_out.commitment_loss
        vae_state.embedding_loss = old_embed_loss + quantizer_out.embedding_loss
        vae_state.soft_codes.append(quantizer_out.soft_codes)

        return self.quantizers[block_idx + 1].upsample_codes(quantizer_out.hard_codes), cross_attn_kv

    def train_or_val_step(self, batch: Dict[str, Tensor], stage: str) -> Dict[str, Tensor]:
        vae_state = self.forward(batch)

        # Convert these sums into means over codebooks. Within each codebook we compute the commitment and
        # embedding losses as the *mean* squared distance between the encoder output and the corresponding code in the
        # codebook, and so in order to be consistent we need to mean across codebooks as well.
        commitment_loss = vae_state.commitment_loss / len(self.quantizers)
        embedding_loss = vae_state.embedding_loss / len(self.quantizers)

        nll, ppl = self.stats_from_logits(vae_state.logits, batch['token_ids'], word_counts = batch['num_words'])
        self.log('nll', nll, prog_bar=True, logger=False, on_step=True, on_epoch=False)

        log_prefix = stage + '_'
        self.log_dict({
            log_prefix + 'nll': nll,
            log_prefix + 'ppl': ppl,
            log_prefix + 'commitment_loss': commitment_loss,
            log_prefix + 'embedding_loss': embedding_loss
        })
        self.log_dict(self.used_code_info_dict())

        logits = vae_state.logits.detach()
        if stage == 'val':
            # BLEU is expensive to compute so we only do it during validation
            reconstructions = self.tokenizer.decode_batch(logits.argmax(dim=-1).tolist())
            reconstructed_words = [sample.split() for sample in reconstructions]

            ground_truth = self.tokenizer.decode_batch(batch['token_ids'].tolist())
            real_words = [[sample.split()] for sample in ground_truth]

            self.log('val_bleu', bleu_score(reconstructed_words, real_words))

        elif stage == 'train':
            loss = nll + embedding_loss + self.hparams.beta * commitment_loss
            if not loss.isfinite():
                self.print("Warning: Got NaN loss. Skipping parameter update.")
                return None

            return {'logits': logits, 'loss': loss}

    def training_step(self, batch: Dict[str, PaddedTensor], batch_index: int, **kwargs) -> Dict[str, Tensor]:
        return self.train_or_val_step(batch, stage='train')

    def validation_step(self, batch: Dict[str, PaddedTensor], batch_index: int):
        self.train_or_val_step(batch, stage='val')

    # Return the quantized latent feature maps only
    def predict(self, batch: Dict[str, PaddedTensor], batch_idx: int, dataloader_idx: Optional[int] = None):
        latent_codes = self.forward(batch).code_indices

        result = {'num_tokens': batch['num_tokens']}
        for i, z in enumerate(latent_codes):
            level_name = self.name_for_latent_level(i, len(latent_codes))
            result[level_name] = z.to_dict()

        return result

    def decode_prior_samples(self, samples: List[PaddedTensor]) -> Tensor:
        # Get the embeddings for each of the latent codes
        last_z = self.quantizers[-1].upsample_codes(self.quantizers[-1].lookup_codes(samples[-1]))
        if len(samples) > 1:
            penultimate_z = self.quantizers[-2].upsample_codes(self.quantizers[-2].lookup_codes(samples[-2]))
        else:
            penultimate_z = None

        # Don't modify the Transformer's hidden state, but have it cross-attend to the penultimate latent sequence
        callback = lambda i, x: (x, penultimate_z)
        return self.decoder(last_z, start_block=-1, block_end_callback=callback)

    # We can't sample without a separately trained prior
    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        return None

    # Used for logging to TensorBoard, mainly
    def used_code_info_dict(self):
        if len(self.quantizers) == 1:
            return {'num_used_codes': self.quantizers[0].num_used_codes()}

        info_dict = {}
        for level, quantizer in enumerate(self.quantizers):
            info_dict[f'code_entropy_{level}'] = quantizer.entropy
            info_dict[f'num_used_codes_{level}'] = quantizer.num_used_codes()

        return info_dict

    @staticmethod
    def name_for_latent_level(level: int, num_levels: int):
        assert num_levels > 0

        if num_levels > 3:
            return f'level {level}'
        if level == 0:
            return 'top'
        if level == num_levels - 1:
            return 'bottom'
        if level == 1:
            return 'middle'