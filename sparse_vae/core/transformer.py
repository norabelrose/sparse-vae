from .transformer_layer import TransformerLayer
from torch import nn, Tensor


class Transformer(nn.Module):
    def __init__(self, num_layers: int, vocab_size: int, d_model: int, num_heads: int = None, **kwargs):
        super().__init__()

        num_heads = num_heads or d_model // 64
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(*[
            TransformerLayer(d_model=d_model, num_heads=num_heads, **kwargs) for _ in range(num_layers)
        ])

        output_embedding = nn.Linear(d_model, vocab_size)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            output_embedding
        )
        output_embedding.weight = self.embedding.weight

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.layers(x)
        return self.output_layer(x)
