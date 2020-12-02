import fast_transformers
import torch.nn.functional as F
from .AutoencoderConfig import AutoencoderConfig

class TextDecoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        
        # The architecture allows arbitrary scale factors- although it's unclear how well the model will do
        # if we train it using one tuple of scaling factors and then switch at test time
        self.scale_factors = config.latent_structure.scaling_factors
        
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
        def expand_length(x, scale_factor):
            x = x.transpose(-2, -1) # Text is [batch, len, embedding] but conv1d expects [batch, channels, len]
            x_depth = x.shape[-2]
            
            kernel = torch.ones((x_depth, x_depth, scale_factor)) # Expand length dimension
            x = F.conv_transpose1d(x, kernel, stride=scale_factor)
            
            return x.transpose(-2, 1) # Flip back to [batch, len, embedding]
        
        x = z_1
        if not scaled_inputs:
            # Pure sampling mode, no encoder
            for scale in self.scale_factors:
                x = expand_length(x, scale)
        else:
            # Encoder-decoder mode
            for decoder_rep, combiner, scale in zip(scaled_inputs, self.combiner_cells, self.scale_factors):
                x = expand_length(x, scale)
                params = combiner(x, decoder_rep)
                
        
        x = self.embedding_upscale1(x)
        
        x = expand_length(x, self.scale_factors[1])
        x = self.embedding_upscale2(x)
        
        x = expand_length(x, self.scale_factors[2])
        x = self.embedding_upscale3(x)
        
        return x