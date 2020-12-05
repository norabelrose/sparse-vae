from .AutoencoderConfig import *
from .funnel_transformers.FunnelTransformer import FunnelTransformer

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, funnel_to_use: Optional[FunnelTransformer] = None):
        super().__init__()

        # Autoencoder might have already made one for us
        self.funnel_transformer = funnel_to_use or FunnelTransformer(config.get_funnel_config())
        self.parameter_convolutions = []

        # For each block, we take the activations from the Funnel Transformer and convolve them down to a lower
        # depth (16 dimensions by default). We do this twice, to get both mu and sigma.
        structure = config.latent_structure
        for i in range(len(structure.block_sizes)):
            param_conv = nn.Conv1d(
                in_channels=structure.overt_depth,
                out_channels=structure.latent_depth * 2,
                kernel_size=1
            )
            self.parameter_convolutions.append(param_conv)
            self.funnel_transformer.add_hidden_state_listener(param_conv, block_index=i)

    # Returns last hidden state and a list of mu & sigma tensors from each block
    def forward(self, x) -> Tuple[torch.tensor, List[torch.tensor]]:
        return self.funnel_transformer(x)
