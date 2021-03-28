from functools import lru_cache
from einops import rearrange
from torch import nn, Tensor
import torch


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, causal: bool = False, sparse: bool = False):
        super().__init__()

        self.causal = causal
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "num_heads must divide d_model evenly"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.reset_kv_cache()

        if sparse:
            from deepspeed.ops.sparse_attention import SparseSelfAttention, FixedSparsityConfig
            config = FixedSparsityConfig(num_heads=num_heads, attention='unidirectional' if causal else 'bidirectional',
                                         different_layout_per_head=True, num_different_global_patterns=4)
            self.sparse_attention = SparseSelfAttention(sparsity_config=config, attn_mask_mode='add')
            self.register_buffer('_sparsity_mask', None)
        else:
            self.sparse_attention = None

    def get_sparsity_mask(self, query_index: int, key_length: int) -> Tensor:
        block_size = self.sparse_attention.sparsity_config.block

        # Copy of the block sparse layout, upsampled to [num_heads, num_blocks, max_seq_len] so that we can select
        # only those keys and values that a given query should attend to. Lazily loaded for autoregressive decoding.
        if self._sparsity_mask is None:
            self._sparsity_mask = self.sparse_attention.master_layout.repeat_interleave(block_size, dim=-1).bool()

        return self._sparsity_mask[:, query_index // block_size, :key_length]

    # Mask should be True where you DON'T want to attend.
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        q = q + positional_encodings_like(q, self.cache_index)  # Position-Infused Attention from "Shortformer" paper
        q = self.q_linear(q)

        # This will be True only when we're using cached keys and values with cross attention
        if self.precomputed_kv:
            k, v = self.key_cache, self.value_cache
            self.cache_index += 1

        # Normal case
        else:
            k = k + positional_encodings_like(k, self.cache_index)
            k, v = self.k_linear(k), self.v_linear(v)

            if self.key_cache is not None:
                # We're being fed new keys and values one token at a time- self-attention case
                if k.shape[-2] == 1:
                    self.key_cache[:, self.cache_index] = k.squeeze(-2)
                    self.value_cache[:, self.cache_index] = v.squeeze(-2)
                    self.cache_index += 1

                    k, v = self.key_cache[:, :self.cache_index], self.value_cache[:, :self.cache_index]

                # We're being fed a bunch of keys and values all at once- cross-attention case. We set the
                # cross_attention flag to True to indicate that we have all the keys and values pre-computed and can
                # ignore future k and v arguments passed to forward() until reset_kv_cache() is called.
                else:
                    self.key_cache.copy_(k)
                    self.value_cache.copy_(v)
                    self.precomputed_kv = True
                    self.cache_index += 1

        q, k, v = (rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads) for x in (q, k, v))

        if self.sparse_attention:
            q_len = q.shape[-2]

            # For autoregressive generation, where we process one single query token at a time. SparseSelfAttention
            # doesn't support this out of the box, but it turns out that in the one-query case we can implement
            # sparse attention relatively efficiently (O(n * sqrt(n)) time and space) with pure PyTorch primitives.
            if q_len == 1:
                # Note that these .view() calls will fail if the number of keys attended to for a given query
                # is different across different attention heads
                layout_mask = self.get_sparsity_mask(self.cache_index, k.shape[-2])
                sparse_k = k[:, layout_mask, :].view(*k.shape[0:2], -1, k.shape[-1])
                sparse_v = v[:, layout_mask, :].view(*v.shape[0:2], -1, v.shape[-1])

                scores = q @ sparse_k.transpose(-1, -2) * k.shape[-1] ** -0.5
                output = scores.softmax(dim=-1) @ sparse_v
            else:
                mask = mask * -1e7 if mask is not None else None  # DeepSpeed *adds* this mask to the attn scores
                attn_mask = torch.ones(q_len, q_len, device=q.device, dtype=torch.bool).triu(1) * -1e7 if self.causal else None
                output = self.sparse_attention(q, k, v, attn_mask=attn_mask, key_padding_mask=mask)
        else:
            # [batch, heads, target length, source length]
            scores = q @ k.transpose(-1, -2) * k.shape[-1] ** -0.5

            # Note that we only apply the upper triangular causal mask during training;
            # during autoregressive decoding there's no "right context" to mask out
            mask = mask[:, None, None, :] if mask is not None and mask.ndim == 2 else mask
            if self.causal and self.key_cache is None:
                causal_mask = torch.ones(*scores.shape[-2:], device=scores.device, dtype=torch.bool).triu(1)
                mask = mask | causal_mask if mask is not None else causal_mask

            if mask is not None:
                scores = scores - mask * 1e7

            # We *don't* apply dropout to the scores here because for some reason it led to really terrible
            # text generation samples and, weirdly, severe overfitting
            output = scores.softmax(dim=-1) @ v

        output = rearrange(output, 'b h l d -> b l (h d)')
        output = self.output_linear(output)

        return output

    def prepare_kv_cache(self, batch_size: int, max_length: int, dtype: torch.dtype):
        # Make sure our cache is a multiple of the sparse attention block size, if applicable
        # if self.sparse_attention:
        #     block_size = self.sparse_attention.sparsity_config.block
        #     max_length += block_size - (max_length % block_size)

        device = self.output_linear.weight.device
        self.key_cache = torch.zeros(batch_size, max_length, self.d_model, device=device, dtype=dtype)
        self.value_cache = torch.zeros(batch_size, max_length, self.d_model, device=device, dtype=dtype)

    def reset_kv_cache(self):
        self.key_cache = None
        self.value_cache = None
        self.cache_index = 0
        self.precomputed_kv = False


def positional_encodings_like(x: Tensor, start: int = 0):
    length = x.shape[-2]
    return get_positional_encodings(start, start + length, x.shape[-1], x.device, x.dtype)

@lru_cache(maxsize=10)
def get_positional_encodings(start: int, end: int, d_model: int, device, dtype) -> Tensor:
    d_model_half = d_model // 2
    frequencies = torch.arange(d_model_half, dtype=dtype, device=device)
    periods = 1 / (10000 ** (frequencies / d_model_half))

    position_ids = torch.arange(start, end, 1.0, dtype=dtype, device=device)
    angles = position_ids[:, None] * periods[None]  # noqa; Outer product

    encodings = torch.empty(end - start, d_model, dtype=dtype, device=device)
    encodings[:, ::2] = angles.sin()
    encodings[:, 1::2] = angles.cos()
    return encodings
