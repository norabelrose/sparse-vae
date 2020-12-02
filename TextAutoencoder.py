import torch.nn.functional as F
from torch import nn
from transformers import FunnelTokenizerFast, FunnelBaseModel
from .AutoencoderCells import *
from .AutoencoderConfig import *
from .TextEncoder import TextEncoder
from .TextDecoder import TextDecoder

class TextAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        
        tokenizer_name = config.latent_structure.pretrained_tokenizer_name()
        self.tokenizer = FunnelTokenizerFast.from_pretrained(tokenizer_name)
        
        # If copy_encoder_weights_to_decoder == True, we should load the weights once here and then hand it
        # off to both the encoder and the decoder
        if config.use_pretrained_encoder and config.copy_encoder_weights_to_decoder:
            funnel = FunnelBaseModel(config.get_funnel_config())
        else:
            funnel = None
        
        self.encoder = TextEncoder(config, funnel_to_use = funnel)
        self.decoder = TextDecoder(config, funnel_to_use = funnel)
        
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
    
    def forward(self, x: Union[list[str], tensor[int]]):
        x = self.tokenizer.encode(x)
        
        mu, logvar = self.encoder(x)
        x = self.sample_latent(mu, logvar)
        x = self.decoder(x)
        
        return x, mu, logvar