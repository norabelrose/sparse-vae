import fast_transformers
import torch.nn.functional as F
from Autoencoder import *
from FunnelTransformer import *


class SamplingCell(nn.Module):
    


class Decoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, funnel_to_use = Autoencoder):
        super().__init__()

        self.funnel_transformer = funnel_to_use or FunnelTransformer(config.get_funnel_config())
        
        # Build a linear 'Transformers are RNNs' autoregressive decoder
        if config.use_autoregressive_decoding:
            builder = fast_transformers.RecurrentDecoderBuilder()
            builder.n_layers = 2
            builder.n_heads = 8
            builder.feed_forward_dimensions = 3072
            builder.query_dimensions = 768
            builder.value_dimensions = 768
            builder.dropout = 0.1
            builder.attention_dropout = 0.1
            builder.self_attention_type = 'causal'
            builder.cross_attention_type = 'full'
            self.decoder_transformer = builder.get()
    
    def forward(self, z_1, scaled_inputs: list = None):

        
        return x