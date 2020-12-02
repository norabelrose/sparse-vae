from torch import nn
from .AutoencoderConfig import *
from .funnel_transformers.ops import RelativePositionalAttention


class EncoderLayer(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()

        self.attention = RelativePositionalAttention()


class EncoderBlock(nn.Module):
    def __init__(self, block_index: int, config: AutoencoderConfig):
        super().__init__()

        num_layers = config.latent_structure.block_sizes[block_index]
        self.layers = [EncoderLayer(config) for _ in range(num_layers)]

    #def forward(self):
