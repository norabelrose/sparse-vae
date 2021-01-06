from functools import partial
from numpy import prod
from torch import nn
from torch import Tensor
from .HparamUtils import *
from .funnel_transformers.FunnelTransformer import FunnelTransformer
import math
import torch.nn.functional as F
import pytorch_lightning as pl
import torch


class Autoencoder(pl.LightningModule):
    default_hparams = AttributeDict(
        encoder_hparams=mutate(
            FunnelTransformer.default_hparams,
            block_sizes=(4, 4, 4, 2, 2),    # Number of layers in each encoder block; reversed for the decoder
            scaling_factors=(2, 2, 4, 4)    # How much the hidden state is downsampled between each encoder block
        ),
        latent_depth=16,                        # Depth of the latent tensors (dimensionality per token)
        use_pretrained_encoder=True,
        copy_encoder_weights_to_decoder=True,
        use_autoregressive_decoding=False,

        lr=1e-4,
        warmup_steps=100,
        weight_decay=0.01
    )

    def __init__(self, hparams: MutableMapping[str, Any]):
        super().__init__()

        # save_hyperparameters() stores the kwargs in self.hparams and ensures they are saved to disk during training.
        hparams = merge(self.default_hparams, hparams)
        self.save_hyperparameters(hparams)

        encoder_hparams = hparams.encoder_hparams
        decoder_hparams = mutate(
            encoder_hparams,
            block_sizes=encoder_hparams.block_sizes[::-1],          # Reverse the order of the blocks
            scaling_factors=encoder_hparams.scaling_factors[::-1],
            upsampling=True
        )
        encoder_hparams = mutate(encoder_hparams, return_block_outputs=True)

        self.encoder_funnel = FunnelTransformer(encoder_hparams)
        self.decoder_funnel = FunnelTransformer(decoder_hparams)

        # Initial input into the decoder
        self.decoder_seed = nn.Parameter(torch.zeros(1, 1, hparams.d_model))
        overt_depth, latent_depth = hparams.latent_depth, hparams.overt_depth

        self.decoder_cells = nn.ModuleList(
            nn.ModuleDict(dict(
                prior=nn.Conv1d(
                    in_channels=overt_depth,
                    out_channels=latent_depth * 2 + overt_depth,
                    kernel_size=1
                ),
                q_of_z_given_x=nn.Conv1d(
                    in_channels=overt_depth * 2,    # Because we concatenate the encoder and decoder states depthwise
                    out_channels=latent_depth * 2,  # Because we need both mu and log sigma
                    kernel_size=1
                ),
                z_upsample=nn.Conv1d(
                    in_channels=latent_depth,
                    out_channels=overt_depth,
                    kernel_size=1
                ),
            ))
            for _ in sum(encoder_hparams.block_sizes)
        )

        # Helpful for saving state during a forward pass
        self.encoder_states = []
        self.kl_divergences = []

        # After each layer in the decoder, call decoder_forward with the layer's output and the corresponding
        # ModuleDict with the corresponding Conv1d modules
        cell_iter = iter(self.decoder_cells)
        for block_idx, block in enumerate(self.decoder_funnel.blocks):
            for layer in block.layers:
                layer.output_transform = partial(self.decoder_layer_forward, next(cell_iter), block_idx)

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

    def sample(self, max_length: int, count: int = 1, temperature: float = 1.0, top_k: int = 1) -> Tensor:
        # Find the sequence length dimension that the seed should have in order to generate the desired output length
        funnel_hparams = self.hparams.encoder_hparams
        total_scaling = prod(funnel_hparams.scaling_factors)
        seed_length = max_length // total_scaling

        seed = self.decoder_seed.expand(count, seed_length, funnel_hparams.d_model)
        output_logits: Tensor = self.decoder_funnel(seed)['logits']  # (batch, seq_len, vocab_size)
        output_ids = output_logits.topk(top_k, dim=-1)

        return output_ids

    # Called once for each Transformer layer in the decoder
    def decoder_layer_forward(self, cell: nn.ModuleDict, block_index: int, layer_output: Tensor) -> Tensor:
        encoder_state = self.encoder_states[block_index] if self.encoder_states else None
        prior_output = cell['prior'](layer_output)

        # Convnet output has 3 components: mu, log standard deviation, and a bias term to add to the input
        overt_depth, latent_depth = self.hparams.latent_depth, self.hparams.overt_depth
        p_mu = prior_output[:, :, :latent_depth]
        p_logsigma = prior_output[:, :, latent_depth:latent_depth * 2]
        xpp = prior_output[:, :, latent_depth * 2:]
        layer_output += xpp

        @torch.jit.script
        def gaussian_kl_divergence(mu1: Tensor, mu2: Tensor, logsigma1: Tensor, logsigma2: Tensor) -> Tensor:
            term1 = -0.5 + logsigma2 - logsigma1
            term2 = 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)
            return term1 + term2

        @torch.jit.script
        def sample_diagonal_gaussian_variable(mu: Tensor, logsigma: Tensor) -> Tensor:
            noise = torch.empty_like(mu).normal_(0., 1.)
            return torch.exp(logsigma) * noise + mu

        # Sample conditioned on the encoder state (used during training)
        if encoder_state is not None:
            q_input = torch.cat([layer_output, encoder_state], dim=-1)
            q_mu, q_logsigma = cell['q_of_z_given_x'](q_input).chunk(2, dim=-1)

            z = sample_diagonal_gaussian_variable(q_mu, q_logsigma)
            kl = gaussian_kl_divergence(q_mu, p_mu, q_logsigma, p_logsigma)
            self.kl_divergences.append(kl)

        # Sample unconditionally (used during evaluation/generation)
        else:
            z = sample_diagonal_gaussian_variable(p_mu, p_logsigma)

        layer_output += cell['z_upsample'](z)
        return layer_output

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        input_tokens = batch['token_ids']
        nonpadding_mask = batch['nonpadding_mask']    # 1 where tokens are NOT padding, 0 where they are
        self.encoder_states = self(input_tokens)

        seed = self.decoder_seed.expand_as(self.encoder_states[-1])
        output_logits: Tensor = self.decoder_funnel(seed)['logits']    # (batch, seq_len, vocab_size)

        # We have to do it this way because the shapes of the KL div. tensors won't match up between decoder cells.
        # Also, to ignore the padding tokens we sum the nonpadding mask to get the total sequence length in this batch.
        kl_sum = sum(kl.sum() for kl in self.kl_divergences)
        kl_divergence = kl_sum / nonpadding_mask.sum()

        negative_log_likelihood = F.nll_loss(output_logits, target=input_tokens, weight=nonpadding_mask)
        negative_elbo = kl_divergence + negative_log_likelihood

        self.kl_divergences.clear()
        return negative_elbo

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index)

    def validation_epoch_end(self, losses: List[Tensor]) -> dict:
        return {'log': {'val_loss': torch.mean(torch.stack(losses))}}
