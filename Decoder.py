from __future__ import annotations

# import fast_transformers
from torch import nn
from dataclasses import *
from .funnel_transformers import FunnelTransformer
from .Utilities import *
from .AutoencoderConfig import AutoencoderConfig


@dataclass
class DecoderCell(nn.Module):
    latent_depth: int
    overt_depth: int
    encoder_state: Optional[Tensor] = None  # The final hidden state of the corresponding encoder block, if applicable.
    last_kl_divergence: Optional[Tensor] = field(init=False, default=None)

    def __post_init__(self):
        self.prior = nn.Conv1d(
            in_channels=self.overt_depth,
            out_channels=self.latent_depth * 2 + self.overt_depth,
            kernel_size=1
        )
        self.q_of_z_given_x = nn.Conv1d(
            in_channels=self.overt_depth * 2,  # Because we concatenate the encoder and decoder states depthwise
            out_channels=self.latent_depth * 2,  # Because we need both mu and log sigma
            kernel_size=1
        )
        self.z_upsample = nn.Conv1d(
            in_channels=self.latent_depth,
            out_channels=self.overt_depth,
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
            q_input = torch.cat([x, self.encoder_state], axis=-1)
            q_mu, q_logsigma = self.q_of_z_given_x(q_input).chunk(2, dim=-1)

            z = sample_diagonal_gaussian_variable(q_mu, q_logsigma)
            self.last_kl_divergence = gaussian_kl_divergence(q_mu, p_mu, q_logsigma, p_logsigma)

        # Sample unconditionally (used during evaluation/generation)
        else:
            z = sample_diagonal_gaussian_variable(p_mu, p_logsigma)

        x = x + self.z_upsample(z)

        return x

class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, funnel_to_use: FunnelTransformer = None):
        super().__init__()

        self.funnel_transformer = funnel_to_use or FunnelTransformer(config.get_funnel_config())
        self.decoder_cells: List[DecoderCell] = []

        latent_depth = config.latent_depth
        overt_depth = config.overt_depth

        for index, layer in self.funnel_transformer.enumerate_layers():
            new_cell = DecoderCell(latent_depth=latent_depth, overt_depth=overt_depth)

            layer.output_transform = new_cell
            self.decoder_cells.append(new_cell)

    def forward(self, x: Tensor, encoder_states: List[Tensor] = None) -> List[Tensor]:
        # We're sampling conditionally
        if encoder_states:
            # The number of encoder states should be equal to the number of encoder (and decoder) blocks
            assert len(encoder_states) == len(self.funnel_transformer.blocks)

            for block, state in zip(self.funnel_transformer.blocks, encoder_states):
                for layer in block.layers:
                    cell: DecoderCell = layer.output_transform
                    cell.encoder_state = state

        return self.funnel_transformer(x)

    # Returns a list of the KL divergences from the last forward pass
    def get_last_pass_kl_stats(self) -> List[Tensor]:
        return [cell.last_kl_divergence for cell in self.decoder_cells]
