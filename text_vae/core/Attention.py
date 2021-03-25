from einops import rearrange
from torch import nn, Tensor
import torch


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, causal: bool = False, sparse: bool = False, dropout: float = 0.1):
        super().__init__()

        self.causal = causal
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "num_heads must divide d_model evenly"

        self.dropout = nn.Dropout(p=dropout)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.reset_kv_cache()

        if sparse:
            from deepspeed.ops.sparse_attention import SparseSelfAttention, FixedSparsityConfig
            config = FixedSparsityConfig(num_heads=num_heads, attention='unidirectional' if causal else 'bidirectional')
            self.sparse_attention = SparseSelfAttention(sparsity_config=config)
        else:
            self.sparse_attention = None

    # Mask should be True where you DON'T want to attend.
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None):
        # This will be True only when we're using cached keys and values with cross attention
        if self.cache_index == -1:
            k, v = self.key_cache, self.value_cache

        # Normal case
        else:
            k, v = self.k_linear(k), self.v_linear(v)

            if self.key_cache is not None:
                # We're being fed new keys and values one token at a time- self-attention case
                if k.shape[-2] == 1:
                    self.key_cache[:, self.cache_index] = k.squeeze(-2)
                    self.value_cache[:, self.cache_index] = v.squeeze(-2)
                    self.cache_index += 1

                    k, v = self.key_cache[:, :self.cache_index], self.value_cache[:, :self.cache_index]

                # We're being fed a bunch of keys and values all at once- cross-attention case. We set the
                # cache_index variable to -1 to indicate that we have all the keys and values pre-computed and can
                # ignore future k and v arguments passed to forward() until reset_kv_cache() is called.
                else:
                    self.key_cache.copy_(k)
                    self.value_cache.copy_(v)
                    self.cache_index = -1

        q = self.q_linear(q)
        if self.sparse_attention:
            mask = mask * -1e7 if mask is not None else None  # DeepSpeed *adds* this mask to the attn scores
            output = self.sparse_attention(q, k, v, key_padding_mask=mask)
            return self.output_linear(output)

        q, k, v = (rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads) for x in (q, k, v))

        # [batch, heads, target length, source length]
        scores = q @ k.transpose(-1, -2) / k.shape[-1] ** -0.5

        # If mask is 2D we treat it as a padding mask over the *keys*
        mask = mask[:, None, None, :] if mask is not None and mask.ndim == 2 else mask
        if self.causal:
            causal_mask = torch.ones(*scores.shape[-2:], device=scores.device, dtype=torch.bool).triu(1)
            mask = mask | causal_mask if mask is not None else causal_mask

        if mask is not None:
            scores = scores - mask * 1e7

        scores = self.dropout(scores)
        output = scores.softmax(dim=-1) @ v
        output = rearrange(output, 'b h l d -> b l (h d)')
        output = self.output_linear(output)

        return output

    def prepare_kv_cache(self, batch_size: int, max_length: int):
        device = self.output_linear.weight.device
        self.key_cache = torch.zeros(batch_size, max_length, self.d_model, device=device)
        self.value_cache = torch.zeros(batch_size, max_length, self.d_model, device=device)

    def reset_kv_cache(self):
        self.key_cache = None
        self.value_cache = None
        self.cache_index = 0
