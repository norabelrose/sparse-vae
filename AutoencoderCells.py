import torch
import torch.nn.functional as F
from torch import nn
from transformers import PerformerAttention
from typing import *

# Combination strategy basically copied from the NVAE source code; may not be ideal
class CombinerCell(nn.Module):
    def __init__(self, d_model: int = 768):
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size = 1)
    
    def forward(self, encoder_rep, decoder_rep):
        encoder_rep = self.depthwise_conv(encoder_rep)
        return encoder_rep + decoder_rep

class DecoderCell(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        self.attention = PerformerAttention(dim=d_model, n_heads=num_heads)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, scale_factor):
        # Nearest neighbor interpolation, as used in the Very Deep VAEs paper
        upscaled_x = F.interpolate(x, scale_factor = scale_factor)
        
        # Pooled input selects what part of the unpooled input is important to keep
        y = self.attention(query=upscaled_x, key=x, value=upscaled_x)
        y = F.gelu(attn_output)
        
        return self.layer_norm(y) # GELU increases the mean of the input so we should normalize

class EncoderCell(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        self.attention = PerformerAttention(dim=d_model, n_heads=num_heads)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, scale_factor):
        pooled_x = F.avg_pool1d(x, kernel_size=scale_factor)
        
        # Pooled input selects what part of the unpooled input is important to keep
        y = self.attention(query=pooled_x, key=x, value=x)
        y = F.gelu(attn_output)
        
        return self.layer_norm(y) # GELU increases the mean of the input so we should normalize

class SamplingCell(nn.Module):
    def __init__(self, d_model: int):
        self.mu_convolution = nn.Conv1d(d_model, d_model, kernel_size = 1)
        self.logvar_convolution = nn.Conv1d(d_model, d_model, kernel_size = 1)
    
    def forward(self, x, params_only: bool = False):
        mu, logvar = self.mu_convolution(x), self.logvar_convolution(x)
        
        if params_only:
            return mu, logvar
        
        # Reparametrization trick
        sigma = logvar.mul(0.5).exp_()
        epsilon = torch.empty_like(sigma).normal_()
        z = epsilon.mul(sigma).add_(mu)
        
        return mu, logvar, z