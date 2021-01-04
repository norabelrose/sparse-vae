from torch import nn
from torch import Tensor
from .HparamUtils import *
from .funnel_transformers.FunnelTransformer import FunnelTransformer
import math
import pytorch_lightning as pl
import torch


def sample_diagonal_gaussian_variable(mu: Tensor, logsigma: Tensor) -> Tensor:
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


class Autoencoder(pl.LightningModule):
    default_hparams = AttributeDict(
        block_sizes=(4, 4, 4, 2, 2),            # Number of layers in each encoder block; reversed for the decoder
        scaling_factors=(2, 2, 4, 4),           # How much the hidden state is downsampled between each encoder block
        d_model=768,                            # Embedding dimension
        latent_depth=16,                        # Depth of the latent tensors (dimensionality per token)
        attention_head_depth=64,                # The dimensionality of each attention head
        use_pretrained_encoder=True,
        copy_encoder_weights_to_decoder=True,
        use_autoregressive_decoding=False,
        use_performer_attention=False,          # See "Rethinking Attention with Performers" paper
        max_sequence_length=512,
        attention_dropout=0.1,

        lr=1e-4,
        warmup_steps=100,
        weight_decay=0.01
    )

    def __init__(self, hparams: Mapping[str, Any]):
        super().__init__()

        # save_hyperparameters() stores the kwargs in self.hparams and ensures they are saved to disk during training.
        hparams = merge(self.default_hparams, hparams)
        self.save_hyperparameters(hparams)

        funnel_hparams = AttributeDict(
            use_performer_attention=hparams.use_performer_attention,
            attention_type='factorized' if hparams.use_performer_attention else 'rel_shift',
            block_sizes=hparams.block_sizes[0:3],
            max_position_embeddings=hparams.max_sequence_length,
            return_block_outputs=True
        )
        self.encoder_funnel = FunnelTransformer(funnel_hparams)
        self.decoder_funnel = FunnelTransformer(funnel_hparams)

        if hparams.use_pretrained_encoder:
            self.encoder_funnel.load_pretrained_weights()

            if hparams.copy_encoder_weights_to_decoder:
                blocks_to_reset = range(3, len(hparams.block_sizes) - 3)
                decoder_funnel = self.encoder_funnel.inverted_copy(reinitialize_blocks=blocks_to_reset)

        self.decoder_cells: List[DecoderCell] = []

        latent_depth = hparams.latent_depth
        overt_depth = hparams.overt_depth

        for index, layer in self.funnel_transformer.enumerate_layers():
            new_cell = DecoderCell(latent_depth=latent_depth, overt_depth=overt_depth)

            layer.output_transform = new_cell
            self.decoder_cells.append(new_cell)

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

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        states = self(batch['text'])

        for block, state in zip(self.decoder_funnel.blocks, states):
            for layer in block.layers:
                cell: DecoderCell = layer.output_transform
                cell.encoder_state = state

        output = self.decoder_funnel(states[-1])

        kl_stats = [cell.last_kl_divergence for cell in self.decoder_cells]
        # TODO: Finish this function

    # Returns the loss
    def validation_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        return self.training_step(batch, batch_index)

    def validation_epoch_end(self, losses: List[Tensor]):
        self.log('val_loss', torch.mean(torch.stack(losses)))


class DecoderCell(nn.Module):
    def __init__(self, latent_depth: int, overt_depth: int):
        super(DecoderCell, self).__init__()

        self.encoder_state: Optional[Tensor] = None  # The final hidden state of the corresponding encoder block
        self.last_kl_divergence: Optional[Tensor] = None
        self.latent_depth, self.overt_depth = latent_depth, overt_depth

        self.prior = nn.Conv1d(
            in_channels=overt_depth,
            out_channels=latent_depth * 2 + overt_depth,
            kernel_size=1
        )
        self.q_of_z_given_x = nn.Conv1d(
            in_channels=overt_depth * 2,  # Because we concatenate the encoder and decoder states depthwise
            out_channels=latent_depth * 2,  # Because we need both mu and log sigma
            kernel_size=1
        )
        self.z_upsample = nn.Conv1d(
            in_channels=latent_depth,
            out_channels=overt_depth,
            kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        prior_output = self.prior(x)
        p_mu = prior_output[:, :self.latent_depth, ...]
        p_logsigma = prior_output[:, self.latent_depth:self.latent_depth * 2, ...]
        xpp = prior_output[:, self.latent_depth * 2:, ...]
        x = x + xpp

        # Sample conditioned on the encoder state (used during training)
        if self.encoder_state:
            q_input = torch.cat([x, self.encoder_state], dim=-1)
            q_mu, q_logsigma = self.q_of_z_given_x(q_input).chunk(2, dim=-1)

            z = sample_diagonal_gaussian_variable(q_mu, q_logsigma)
            self.last_kl_divergence = gaussian_kl_divergence(q_mu, p_mu, q_logsigma, p_logsigma)

        # Sample unconditionally (used during evaluation/generation)
        else:
            z = sample_diagonal_gaussian_variable(p_mu, p_logsigma)

        x = x + self.z_upsample(z)

        return x
