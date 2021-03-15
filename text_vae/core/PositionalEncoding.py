from torch import nn, Tensor
from .Transformers import get_positional_encodings
import torch


# Convenience class for learned absolute positional encodings
class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        super().__init__()

        initial_encodings = get_positional_encodings(max_length, d_model, device=None, dtype=torch.float32)
        self.weight = nn.Parameter(initial_encodings)

    def forward(self, x: Tensor) -> Tensor:
        x_len = x.shape[-2]
        return x + self.weight[:x_len]
