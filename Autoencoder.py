from torch import nn
from typing import *
from .AutoencoderConfig import AutoencoderConfig
from .Decoder import Decoder
from .funnel_transformers.FunnelTransformer import FunnelTransformer
from .PretrainedModelManager import PretrainedModelManager


class Autoencoder(nn.Module):
    def __init__(self, config: Optional[AutoencoderConfig] = None):
        super().__init__()

        config = config or AutoencoderConfig.default()  # Just use default values if we aren't given a config object

        # If copy_encoder_weights_to_decoder == True, we should load the weights once here and then hand it
        # off to both the encoder and the decoder
        decoder_funnel = None
        if config.use_pretrained_encoder:
            self.encoder_funnel = PretrainedModelManager.get_model(config.block_sizes)
            if config.copy_encoder_weights_to_decoder:
                blocks_to_reset = range(3, len(config.block_sizes) - 3)
                decoder_funnel = self.encoder_funnel.inverted_copy(reinitialize_blocks=blocks_to_reset)
        else:
            self.encoder_funnel = FunnelTransformer(config.get_funnel_config())

        self.decoder = Decoder(config, funnel_to_use = decoder_funnel)
