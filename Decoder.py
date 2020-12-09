from __future__ import annotations

# import fast_transformers
from torch import nn
from dataclasses import *
from .funnel_transformers import FunnelTransformer
from .Utilities import *
if TYPE_CHECKING:   # Avoid circular dependency
    from .Autoencoder import Autoencoder, AutoencoderConfig


@dataclass
class DecoderCell(nn.Module):
    latent_depth: int
    overt_depth: int
    encoder_state: Optional[Tensor] = None  # The final hidden state of the corresponding encoder block, if applicable.
    kl_divergence: Optional[Tensor] = field(init=False, default=None)

    def __post_init__(self):
        self.prior = nn.Conv1d(
            in_channels=self.overt_depth,
            out_channels=self.latent_depth * 2 + self.overt_depth,
            kernel_size=1
        )
        self.q_param_convolution = nn.Conv1d(
            in_channels=self.overt_depth * 2,  # Because we concatenate the encoder and decoder states depthwise
            out_channels=self.latent_depth * 2,  # Because we need both mu and log sigma
            kernel_size=1
        )
        self.z_upsample = nn.Conv1d(
            in_channels=self.latent_depth,
            out_channels=self.overt_depth,
            kernel_size=1
        )

    # Returns the KL divergence from the last forward pass and then resets in preparation for the next pass.
    def pop_last_kl_divergence(self) -> Optional[Tensor]:
        last_kl = self.kl_divergence
        self.kl_divergence = None

        return last_kl

    def sample(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        prior_output = self.prior(x)
        p_mu = prior_output[:, :self.latent_depth, ...]
        p_logsigma = prior_output[:, self.latent_depth:self.latent_depth * 2, ...]
        xpp = prior_output[:, self.latent_depth * 2:, ...]
        x = x + xpp

        # Sample conditioned on the encoder state (used during training)
        if self.encoder_state:
            q_input = torch.cat([x, self.encoder_state], axis=-1)
            q_mu, q_logsigma = self.q_param_convolution(q_input).chunk(2, dim=-1)

            z = sample_diagonal_gaussian_variable(q_mu, q_logsigma)
            self.kl_divergence = gaussian_kl_divergence(q_mu, p_mu, q_logsigma, p_logsigma)

        # Sample unconditionally (used during evaluation/generation)
        else:
            z = sample_diagonal_gaussian_variable(p_mu, p_logsigma)

        return x, z

    def forward(self, x: Tensor) -> Tensor:
        x, z = self.sample(x)
        x = x + self.z_upsample(z)

        return x

class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, funnel_to_use: FunnelTransformer = None):
        super().__init__()

        self.funnel_transformer = funnel_to_use or FunnelTransformer(config.get_funnel_config())
        self.decoder_cells = []

        latent_depth = config.latent_depth
        overt_depth = config.overt_depth

        for index, layer in self.funnel_transformer.enumerate_layers():
            new_cell = DecoderCell(latent_depth=latent_depth, overt_depth=overt_depth)

            layer.output_transform = new_cell
            self.decoder_cells.append(new_cell)

    def forward(self, x: Tensor, encoder_states: List[Tensor] = None) -> Union[Tensor, List[Tensor]]:
        # We're sampling conditionally
        if encoder_states:
            # The number of encoder states should be equal to the number of encoder (and decoder) blocks
            assert len(encoder_states) == len(self.funnel_transformer.blocks)

            for block, state in zip(self.funnel_transformer.blocks, encoder_states):
                for layer in block.layers:
                    cell: DecoderCell = layer.output_transform
                    cell.encoder_state = state

            output = self.funnel_transformer(x)
            kl_stats = [cell.pop_last_kl_divergence() for cell in self.decoder_cells]
            return output, kl_stats

        return self.funnel_transformer(x)
