from argparse import ArgumentParser
from torch import nn
from .Utilities import *
from .funnel_transformers.FunnelTransformer import FunnelTransformer
import pytorch_lightning as pl
import torch


class Autoencoder(pl.LightningModule):
    default_hparams = dict(
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

        learning_rate=1e-4,
        weight_decay=0.01
    )

    # For the command line interface
    @classmethod
    def add_model_specific_args(cls, parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for param, default in cls.default_hparams:
            # For block_sizes and scaling_factors
            if isinstance(default, Sequence):
                parser.add_argument("--" + param, nargs='+', type=type(default[0]))
            else:
                parser.add_argument("--" + param, type=type(default), default=default)

        return parser

    def __init__(self, **kwargs):
        super().__init__()

        # save_hyperparameters() stores the kwargs in self.hparams and ensures they are saved to disk during training.
        kwargs = {**self.default_hparams, **kwargs}
        self.save_hyperparameters(kwargs)

        funnel_hparams = transmute(
            self.hparams,
            'use_performer_attention',
            attention_type="'factorized' if use_performer_attention else 'rel_shift'",
            block_sizes='block_sizes[0:3]',
            max_position_embeddings='max_sequence_length',
            return_block_outputs=True
        )
        self.encoder_funnel = FunnelTransformer(**funnel_hparams)
        self.decoder_funnel = FunnelTransformer(**funnel_hparams)

        if self.hparams.use_pretrained_encoder:
            self.encoder_funnel.load_pretrained_weights()

            if self.hparams.copy_encoder_weights_to_decoder:
                blocks_to_reset = range(3, len(self.hparams.block_sizes) - 3)
                decoder_funnel = self.encoder_funnel.inverted_copy(reinitialize_blocks=blocks_to_reset)

        self.decoder_cells: List[DecoderCell] = []

        latent_depth = self.hparams.latent_depth
        overt_depth = self.hparams.overt_depth

        for index, layer in self.funnel_transformer.enumerate_layers():
            new_cell = DecoderCell(latent_depth=latent_depth, overt_depth=overt_depth)

            layer.output_transform = new_cell
            self.decoder_cells.append(new_cell)

    def configure_optimizers(self):
        # TODO: Add LR decay
        return torch.optim.AdamW(**transmute(self.hparams, 'weight_decay', lr='learning_rate'),
                                 params=self.parameters())

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
