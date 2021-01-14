from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from numpy import prod
from omegaconf import OmegaConf
from torch import nn
from torch import Tensor
from .HparamUtils import *
from .FunnelTransformer import FunnelTransformer, FunnelTransformerHparams
import math
import torch.nn.functional as F
import pytorch_lightning as pl
import torch


@dataclass
class AutoencoderHparams:
    encoder: FunnelTransformerHparams = FunnelTransformerHparams(
        block_sizes=(4, 4, 4, 2, 2),  # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(2, 2, 4, 4)  # How much the hidden state is downsampled between each encoder block
    )
    latent_depth: int = 16  # Depth of the latent tensors (dimensionality per token)
    use_pretrained_encoder: bool = True
    copy_encoder_weights_to_decoder: bool = True
    use_autoregressive_decoding: bool = False

    grad_clip_threshold: float = 200.0
    grad_skip_threshold: float = 400.0
    lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams: OmegaConf):
        super().__init__()

        # save_hyperparameters() stores the hparams in self.hparams and ensures they are saved to disk during training.
        self.save_hyperparameters(hparams)

        encoder_hparams = hparams.encoder
        decoder_hparams = mutate(
            encoder_hparams,
            block_sizes=encoder_hparams.block_sizes[::-1],          # Reverse the order of the blocks
            scaling_factors=encoder_hparams.scaling_factors[::-1],
            upsampling=True
        )
        all_blocks = list(range(len(encoder_hparams.block_sizes)))
        encoder_hparams = mutate(encoder_hparams, return_block_outputs=all_blocks)

        # This way, the encoder and decoder Transformers share information about, i.e., the padding mask
        self.encoder_funnel = FunnelTransformer(encoder_hparams)
        attn_state = self.encoder_funnel.attention_state
        attn_state.cache_masks = True
        self.decoder_funnel = FunnelTransformer(decoder_hparams, shared_attention_state=attn_state)

        # Initial input into the decoder
        overt_depth, latent_depth = encoder_hparams.d_model, hparams.latent_depth
        self.decoder_seed = nn.Parameter(torch.zeros(1, 1, overt_depth))

        # Construct the decoder cells which generate p(z), q(z|x), and p(x|z) given the output of a Transformer layer
        def linear_with_gelu(input_dim, output_dim):
            # Note that we apply the activation function FIRST and then the Linear layer
            return nn.Sequential(nn.GELU(), nn.Linear(input_dim, output_dim))

        self.decoder_cells = nn.ModuleList([
            nn.ModuleDict({
                # Output contains mu & log sigma for p(z)
                'p(z)': linear_with_gelu(overt_depth, latent_depth * 2),
                # Input is the encoder and decoder states concatenated depthwise, output is mu & log sigma for q(z|x)
                'q(z|x)': linear_with_gelu(overt_depth * 2, latent_depth * 2),
                'p(x|z)': linear_with_gelu(latent_depth, overt_depth)
            })
            for _ in range(sum(encoder_hparams.block_sizes))
        ])

        # After each layer in the decoder, call decoder_layer_forward with the layer's output and the block and
        # layer indices
        absolute_idx = 0
        for block_idx, block in enumerate(self.decoder_funnel.blocks):
            for layer in block.layers:
                layer.output_transform = partial(self.decoder_layer_forward, absolute_idx, block_idx)
                absolute_idx += 1

        # Load pretrained weights
        if hparams.use_pretrained_encoder:
            self.encoder_funnel.load_pretrained_weights()

            if hparams.copy_encoder_weights_to_decoder:
                # Only copy the first three blocks of the encoder because they are the ones that are pretrained
                encoder_blocks = self.encoder_funnel.blocks[:3]
                decoder_blocks = self.decoder_funnel.blocks[-3:]

                for enc_block, dec_block in zip(encoder_blocks, reversed(decoder_blocks)):
                    dec_block.load_state_dict(enc_block.state_dict())

    def configure_optimizers(self):
        adam = torch.optim.AdamW(**select(self.hparams, 'weight_decay', 'lr'), params=self.parameters())

        # Cosine decay learning rate schedule with warmup steps
        def cosine_with_warmup(current_step, num_cycles=1):
            warmups = self.hparams.warmup_steps
            if current_step < warmups:
                return float(current_step) / float(max(1, warmups))

            total_steps = self.trainer.max_steps
            assert total_steps, "Max training steps must be known to use lr decay."

            progress = float(current_step - warmups) / float(max(1, total_steps - warmups))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(adam, cosine_with_warmup)
        return [adam], [scheduler]

    # Returns hidden states of the encoder
    def forward(self, batch: Tensor) -> List[Tensor]:
        return self.encoder_funnel(batch)

    def sample(self, max_length: int, count: int = 1, top_k: int = 1, return_latents: bool = False,
               clamped_latents: Optional[Mapping[int, Tensor]] = None,
               temperature: Union[float, Mapping[int, float]] = 1.0) -> Tensor:
        # Allow for style transfer-type experiments
        if clamped_latents:
            self.clamped_latents = clamped_latents

        # The temperature parameter can either be a Mapping that assigns different temperatures to different layer
        # indices, or a single float for all layers
        if isinstance(temperature, Mapping):
            self.temperatures = defaultdict(lambda: 1.0)
            self.temperatures.update(temperature)
        elif isinstance(temperature, float):
            self.temperatures = defaultdict(lambda: temperature)

        # Find the sequence length dimension that the seed should have in order to generate the desired output length
        funnel_hparams = self.hparams.encoder_hparams
        total_scaling = prod(funnel_hparams.scaling_factors)
        seed_length = max_length // total_scaling

        seed = self.decoder_seed.expand(count, seed_length, funnel_hparams.d_model)
        output_logits: Tensor = self.decoder_funnel(seed)['logits']  # (batch, seq_len, vocab_size)
        output_ids = output_logits.topk(top_k, dim=-1)

        self.clamped_latents = {}
        self.temperatures = defaultdict(lambda: 1.0)
        return output_ids

    # Called once for each Transformer layer in the decoder
    def decoder_layer_forward(self, layer_index: int, block_index: int, layer_output: Tensor) -> Tensor:
        cell = self.decoder_cells[layer_index]
        p_mu, p_logsigma = cell['p(z)'](layer_output).chunk(2, dim=-1)  # Prior

        @torch.jit.script
        def gaussian_kl_divergence(mu1: Tensor, mu2: Tensor, logsigma1: Tensor, logsigma2: Tensor) -> Tensor:
            term1 = -0.5 + logsigma2 - logsigma1
            term2 = 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)
            return term1 + term2

        @torch.jit.script
        def sample_diagonal_gaussian_variable(mu: Tensor, logsigma: Tensor, temperature: float = 1.0) -> Tensor:
            noise = torch.empty_like(mu).normal_(0., 1.)    # Reparameterization trick
            stddev = logsigma.exp()
            if temperature != 1.0:
                stddev *= temperature

            return stddev * noise + mu

        # Sample conditioned on the encoder state (used during training)
        if self.encoder_states:
            encoder_state = self.encoder_states[block_index]

            q_input = torch.cat([layer_output, encoder_state], dim=-1)
            q_mu, q_logsigma = cell['q(z|x)'](q_input).chunk(2, dim=-1)

            kl_tensor = gaussian_kl_divergence(q_mu, p_mu, q_logsigma, p_logsigma)
            z = sample_diagonal_gaussian_variable(q_mu, q_logsigma)

            # Gives us the appropriately scaled mask for the current block
            padding_mask = self.decoder_funnel.attention_state.input_mask
            kl_tensor.masked_fill_(padding_mask, 0.0)       # Ignore KL divergences for padding positions

            self.total_kl_divergence += kl_tensor.sum()
            self.total_nonpadding_positions += (~padding_mask).sum()

        # Sample unconditionally (used during evaluation/generation)
        elif (z := self.clamped_latents.get(layer_index)) is None:
            z = sample_diagonal_gaussian_variable(p_mu, p_logsigma, self.temperatures[layer_index])

        layer_output += cell['p(x|z)'](z)
        return layer_output

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Any]:
        input_tokens, padding_mask = batch['token_ids'], batch['padding_mask']
        self.encoder_states = self(input_tokens)

        # These attributes will be updated on each pass of decoder_layer_forward()
        device = input_tokens.device
        self.total_kl_divergence = torch.tensor(0.0, device=device)
        self.total_nonpadding_positions = torch.tensor(0, device=device)

        seed = self.decoder_seed.expand_as(self.encoder_states[-1])
        output_logits: Tensor = self.decoder_funnel(seed)['logits']    # (batch, seq_len, vocab_size)
        self.encoder_states = None

        # We have to be careful to exclude padding positions from our KL divergence calculation
        kl_divergence = self.total_kl_divergence / (self.total_nonpadding_positions * self.hparams.latent_depth)
        negative_log_likelihood = F.nll_loss(output_logits, target=input_tokens, weight=~padding_mask)
        self.log_dict({'kl': kl_divergence, 'nll': negative_log_likelihood})

        negative_elbo = kl_divergence + negative_log_likelihood
        return {'loss': negative_elbo}

    # Implements gradient skipping to stabilize training as described in 'Very Deep VAEs'
    def on_after_backward(self):
        # Don't clip at all if .grad_clip_threshold is falsy
        if not (clip_threshold := self.hparams.grad_clip_threshold):
            return

        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip_threshold)
        if grad_norm > self.hparams.grad_skip_threshold:
            self.zero_grad()

        self.log('grad_norm', grad_norm)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Dict[str, Any]:
        return self.training_step(batch, batch_index)

    def validation_epoch_end(self, losses: List[Tensor]):
        self.log('val_loss', torch.mean(torch.stack(losses)))
