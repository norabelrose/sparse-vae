import torch
from torch import nn
from typing import *
from AutoencoderConfig import *
from Encoder import Encoder
from Decoder import Decoder
from PretrainedModelManager import PretrainedModelManager


class Autoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()

        structure = config.latent_structure

        # If copy_encoder_weights_to_decoder == True, we should load the weights once here and then hand it
        # off to both the encoder and the decoder
        encoder_funnel = None
        decoder_funnel = None
        if config.use_pretrained_encoder:
            encoder_funnel = PretrainedModelManager.get_model(structure.block_sizes)
            if config.copy_encoder_weights_to_decoder:
                blocks_to_reset = range(3, len(structure.block_sizes) - 3)
                decoder_funnel = encoder_funnel.inverted_copy(reinitialize_blocks=blocks_to_reset)

        self.encoder = Encoder(config, funnel_to_use = encoder_funnel)
        self.decoder = Decoder(config, funnel_to_use = decoder_funnel)

        #num_cells = len(scale_factors) if not tie_weights else 1
        #self.combiner_cells = [CombinerCell(768, 12) for _ in range(num_cells)]

    def sample_latent(self, mu, logvar):
        # Apparently the whole stochasticity thing is just a training regularization
        if self.training:
            std = logvar.mul(0.5).exp_() # Convert from the log variance to the stddev in-place
            epsilon = torch.empty_like(x).normal_()

            # Reparameterization trick
            return epsilon.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        x, activations = self.encoder(x)
        x = self.sample_latent(mu, logvar)
        x = self.decoder(x)

        return x, mu, logvar