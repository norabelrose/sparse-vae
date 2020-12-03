from .AutoencoderConfig import *
from .funnel_transformers.modeling import FunnelTransformer

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, funnel_to_use: Optional[FunnelTransformer] = None):
        super().__init__()

        # Autoencoder might have already made one for us
        self.funnel_transformer = funnel_to_use or FunnelTransformer(config.get_funnel_config())

        structure = config.latent_structure
        scales = structure.scaling_factors

        # For each scale, we take the activations from the Funnel Transformer and convolve them down to a lower
        # depth (16 dimensions by default). We do this twice, to get both mu and sigma.
        self.parameter_convolutions = [nn.Conv1d(
            in_channels=structure.overt_depth,
            out_channels=structure.latent_depth * 2,
            kernel_size=1
        ) for _ in scales]

        self.low_res_scaling_factors = scales[3:]

    # Returns mu, sigma, and a list of all the scaled inputs
    def forward(self, x) -> Tuple[torch.tensor, torch.tensor, List[torch.tensor]]:
        # Collect the activations from the end of each block
        activations = self.funnel_transformer(x)

        # List which will store the input scaled down at each successive scale
        scaled_inputs = []

        num_cells = len(self.cells)
        for i, scale in enumerate(self.scale_factors):
            cell = self.cells[i % num_cells]

            x = cell(x, scale)
            scaled_inputs.append(x)

        mu, logvar = self.z0_sampler(x, params_only=True)
        return mu, logvar, scaled_inputs
