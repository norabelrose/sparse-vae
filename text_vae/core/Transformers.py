from functools import lru_cache
from typing import Callable, Optional

import torch
from torch import nn, Tensor

from .GenerationUtils import GenerationStrategy, decode_next_token_with_context


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, causal: bool = False, use_cross_attention: bool = False):
        super(TransformerLayer, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=0.1)
        if use_cross_attention:
            # self.alpha = nn.Parameter(torch.tensor(1.0))
            self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=0.1)
            self.cross_attn_layer_norm = nn.LayerNorm(d_model)
        else:
            self.cross_attention = None
            self.cross_attn_layer_norm = None

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False)  # Superfluous since this is followed by layer norm
        )

        self.causal = causal
        self.dropout = nn.Dropout(p=0.1)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn_layer_norm = nn.LayerNorm(d_model)

    def forward(self, qkv: Tensor, cross_attn_kv: Tensor = None, padding_mask: Tensor = None) -> Tensor:
        if self.causal:
            causal_mask = torch.ones(qkv.shape[-2], qkv.shape[-2], device=qkv.device, dtype=torch.bool).triu(1)
        else:
            causal_mask = None

        qkv = qkv.movedim(0, 1)
        y, _ = self.attention(qkv, qkv, qkv, attn_mask=causal_mask,
                              key_padding_mask=padding_mask.bool() if padding_mask is not None else None,
                              need_weights=False)
        qkv = y = self.attn_layer_norm(qkv + y)

        if self.cross_attention:
            cross_attn_kv = cross_attn_kv.movedim(0, 1)
            qkv, _ = self.cross_attention(qkv, cross_attn_kv, cross_attn_kv, need_weights=False)
            qkv = y = y + self.cross_attn_layer_norm(qkv)  # self.alpha *

        y = self.ffn(y)
        y = self.dropout(y)
        y = self.ffn_layer_norm(qkv + y)

        y = y.movedim(1, 0)
        return y


def positional_encodings_like(x: Tensor):
    return get_positional_encodings(x.shape[1], x.shape[2], x.device, x.dtype)

@lru_cache(maxsize=10)
def get_positional_encodings(seq_len: int, d_model: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    d_model_half = d_model // 2
    frequencies = torch.arange(d_model_half, dtype=dtype, device=device)
    periods = 1 / (10000 ** (frequencies / d_model_half))

    position_ids = torch.arange(0, seq_len, 1.0, dtype=dtype, device=device)
    angles = position_ids[:, None] * periods[None]  # noqa; Outer product

    encodings = torch.empty(seq_len, d_model, dtype=dtype, device=device)
    encodings[:, ::2] = angles.sin()
    encodings[:, 1::2] = angles.cos()
    return encodings * (d_model ** -0.5)  # noqa


@torch.no_grad()
def autoregressive_decode_transformer(
    strategy: GenerationStrategy,
    transformer_callable: Callable,
    logit_callable: Callable,           # Upsamples from hidden states to logits
    embedding: nn.Module,
    start_symbol: int,                  # [CLS] in the tokenizer we're using
    end_symbol: int,                    # [SEP] in the tokenizer we're using
    context: Optional[Tensor] = None,   # Keys/values for cross attention
    min_length: int = 10,               # Will never emit [SEP] while the current length < min_length
    max_length: int = 200,
    k: int = 10,                        # The beam size for beam search and top K sampling

    num_samples: int = 1,
    start_embedding: Optional[Tensor] = None,
    d_model: int = None
):
    device = next(embedding.parameters()).device
    start_symbol = torch.tensor(start_symbol, device=device)
    stop_symbol = torch.tensor(end_symbol, device=device)

    live_sample_mask = torch.ones(num_samples, device=device, dtype=torch.bool)

    d_model = d_model or embedding.weight.shape[-1]
    out_embeds = get_positional_encodings(max_length, d_model, device, dtype=torch.float32)
    out_embeds = out_embeds.unsqueeze(0).expand(num_samples, *out_embeds.shape)
    out_embeds[:, 0] += start_embedding if start_embedding is not None else embedding(start_symbol)

    output_ids = torch.zeros(num_samples, max_length, device=device, dtype=torch.long)
    output_ids[:, 0] = start_symbol

    assert strategy != GenerationStrategy.Beam

    for current_idx in range(1, max_length):
        ctx = context[:, :current_idx] if context is not None else None
        next_output = transformer_callable(out_embeds[:, :current_idx], ctx)[:, -1]
        next_logits = logit_callable(next_output)

        # Make the end symbol infinitely unlikely if we're not at the min length yet
        if current_idx < min_length:
            next_logits[:, end_symbol] = -float('inf')

        next_ids = decode_next_token_with_context(next_logits, strategy, k)

        output_ids[:, current_idx] = next_ids
        out_embeds[:, current_idx] += embedding(next_ids)

        live_sample_mask &= (next_ids != stop_symbol)
        if not live_sample_mask.any():
            output_ids = output_ids[:, :current_idx + 1]  # Get rid of any excess padding
            break

    return output_ids
