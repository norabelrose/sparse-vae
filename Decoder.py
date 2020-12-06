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
    latent_depth: InitVar[int]
    overt_depth: InitVar[int]
    encoder_state: Tensor = None  # The final hidden state of the corresponding encoder block, if applicable.

    def __post_init__(self, latent_depth: int, overt_depth: int):
        self.q_param_convolution = nn.Conv1d(
            in_channels=overt_depth * 2,    # Because we concatenate the encoder and decoder states depthwise
            out_channels=latent_depth * 2,  # Because we need both mu and log sigma
            kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        q_input = torch.cat([x, self.encoder_state], axis=-1)
        q_mu, q_logvar = self.q_param_convolution(q_input).chunk(2, dim=-1)
        z = sample_diagonal_gaussian_variable(q_mu, q_logvar)

class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, funnel_to_use: Autoencoder = None):
        super().__init__()

        self.funnel_transformer = funnel_to_use or FunnelTransformer(config.get_funnel_config())

        for index, layer in self.funnel_transformer.enumerate_layers():
            layer.add_output_transform()
        
        # Build a linear 'Transformers are RNNs' autoregressive decoder
        # if config.use_autoregressive_decoding:
        #     builder = fast_transformers.RecurrentDecoderBuilder()
        #     builder.n_layers = 2
        #     builder.n_heads = 8
        #     builder.feed_forward_dimensions = 3072
        #     builder.query_dimensions = 768
        #     builder.value_dimensions = 768
        #     builder.dropout = 0.1
        #     builder.attention_dropout = 0.1
        #     builder.self_attention_type = 'causal'
        #     builder.cross_attention_type = 'full'
        #     self.decoder_transformer = builder.get()
