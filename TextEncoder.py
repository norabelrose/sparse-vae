from transformers import FunnelBaseModel
from .AutoencoderConfig import *

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, funnel_to_use: Optional[FunnelBaseModel] = None):
        super().__init__()
        
        # TextAutoencoder might have already made one for us
        self.funnel_transformer = funnel_to_use or FunnelBaseModel(config.get_funnel_config())
        self.low_res_scaling_factors = config.latent_structure.scaling_factors[3:]
    
    # Returns mu, sigma, and a list of all the scaled inputs
    def forward(self, x) -> Tuple[torch.tensor, torch.tensor, List[torch.tensor]]:
        # Collect the last hidden state and also the activations from the end of each block
        x, activations = self.funnel_transformer(x, output_hidden_states = True, block_end_hidden_states_only = True)
        
        # List which will store the input scaled down at each successive scale
        scaled_inputs = []
        
        num_cells = len(self.cells)
        for i, scale in enumerate(self.scale_factors):
            cell = self.cells[i % num_cells]
            
            x = cell(x, scale)
            scaled_inputs.append(x)
        
        mu, logvar = self.z0_sampler(x, params_only = True)
        return mu, logvar, scaled_inputs
