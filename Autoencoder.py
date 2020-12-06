import torch
from torch import nn
from typing import *
from .AutoencoderConfig import AutoencoderConfig
from .Decoder import Decoder
from .funnel_transformers.FunnelTransformer import FunnelTransformer
from .PretrainedModelManager import PretrainedModelManager


class Autoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()

        structure = config.latent_structure

        # If copy_encoder_weights_to_decoder == True, we should load the weights once here and then hand it
        # off to both the encoder and the decoder
        decoder_funnel = None
        if config.use_pretrained_encoder:
            self.encoder_funnel = PretrainedModelManager.get_model(structure.block_sizes)
            if config.copy_encoder_weights_to_decoder:
                blocks_to_reset = range(3, len(structure.block_sizes) - 3)
                decoder_funnel = self.encoder_funnel.inverted_copy(reinitialize_blocks=blocks_to_reset)
        else:
            self.encoder_funnel = FunnelTransformer(config.get_funnel_config())

        self.decoder = Decoder(config, funnel_to_use = decoder_funnel)

        #num_cells = len(scale_factors) if not tie_weights else 1
        #self.combiner_cells = [CombinerCell(768, 12) for _ in range(num_cells)]

    def sample_latent(self, mu, logvar):
        # Apparently the whole stochasticity thing is just a training regularization
        if self.training:
            std = logvar.mul(0.5).exp_() # Convert from the log variance to the stddev in-place
            epsilon = torch.empty_like(mu).normal_()

            # Reparameterization trick
            return epsilon.mul(std).add_(mu)
        else:
            return mu

    # def forward(self, x):
    #    activations = self.encoder_funnel(x)
        # x = self.sample_latent(mu, logvar)
        # x = self.decoder(x)

        # return x, mu, logvar
