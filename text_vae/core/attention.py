from contextlib import contextmanager
from functools import lru_cache
from einops import rearrange
from torch import nn, Tensor
from typing import *
from .padded_tensor import PaddedTensor
import torch


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        causal = False,
        sparse = False,
        rel_pos_attn = False,
        learned_queries: int = None
    ):
        super().__init__()

        self.causal = causal
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "num_heads must divide d_model evenly"

        # We can use learned queries to extract a single vector or a fixed size sequence of vectors out of a sequence
        if learned_queries:
            self.learned_queries = nn.Parameter(torch.randn(1, learned_queries, d_model))
        else:
            self.q_linear = nn.Linear(d_model, d_model)
            self.learned_queries = None

        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.rel_pos_attn = rel_pos_attn
        if rel_pos_attn:
            d_head = d_model // num_heads
            self.r_w_bias = nn.Parameter(torch.zeros(num_heads, 1, d_head))
            self.r_r_bias = nn.Parameter(torch.zeros(num_heads, 1, d_head))
            self.r_kernel = nn.Parameter(torch.zeros(num_heads, d_model, d_head))

        self.cache_index = 0
        self.precomputed_kv = False
        self.key_cache = None
        self.value_cache = None

        if sparse:
            from deepspeed.ops.sparse_attention import SparseSelfAttention
            from .sliding_window_sparsity_config import SlidingWindowSparsityConfig

            config = SlidingWindowSparsityConfig(num_heads=num_heads, window_size=2)
            self.sparse_attention = SparseSelfAttention(
                sparsity_config=config,
                attn_mask_mode='add',
                max_seq_length=12_512
            )
        else:
            self.sparse_attention = None

    def forward(self, q: Optional[Tensor], k: PaddedTensor, v: Tensor, pos_enc = None):
        # When using learned queries, we ignore the q argument- it's expected that you'll set that to None
        if self.learned_queries is not None:
            q = self.learned_queries.expand(k.shape[0], *self.learned_queries.shape[1:])
        else:
            # Position-Infused Attention from "Shortformer" paper
            if not self.rel_pos_attn:
                q = q + positional_encodings_like(q, self.cache_index)
            q = self.q_linear(q)

        # This will be True only when we're using cached keys and values with cross attention
        if self.precomputed_kv:
            k, v = self.key_cache, self.value_cache

        # Normal case
        else:
            if not self.rel_pos_attn:
                k = k + positional_encodings_like(k, self.cache_index)

            k, v = self.k_linear(k), self.v_linear(v)
            if self.kv_cache_length:
                k, v = self._update_kv_cache(k, v)

        mask = getattr(k, 'padding', None)
        q, k, v = (rearrange(x, '... l (h d) -> ... h l d', h=self.num_heads) for x in (q, k, v))
        if self.sparse_attention and self.key_cache is None:
            q_len = q.shape[-2]

            mask = mask * -1e7 if mask is not None else None  # DeepSpeed *adds* this mask to the attn scores
            attn_mask = torch.ones(q_len, q_len, device=q.device, dtype=torch.bool).triu(1) * -1e7 if self.causal else None
            output = self.sparse_attention(q, k, v, attn_mask=attn_mask, key_padding_mask=mask)
        else:
            # Apply relative positional attention scores here if needed
            if self.rel_pos_attn:
                assert pos_enc is not None
                scores = (q + self.r_w_bias) @ k.transpose(-1, -2) * k.shape[-1] ** -0.5
                scores = scores + self._rel_pos_attn_term(q, k.shape[-2], pos_enc)
            else:
                scores = q @ k.transpose(-1, -2) * k.shape[-1] ** -0.5

            # Note that we only apply the upper triangular causal mask during training;
            # during autoregressive decoding there's no "right context" to mask out
            mask = mask[..., None, None, :] if mask is not None and mask.ndim >= 2 else mask
            if self.causal and self.key_cache is None:
                causal_mask = torch.ones(*scores.shape[-2:], device=scores.device, dtype=torch.bool).triu(1)
                mask = mask | causal_mask if mask is not None else causal_mask

            if mask is not None:
                scores = scores - mask * 1e7

            output = scores.softmax(dim=-1) @ v

        output = rearrange(output, '... h l d -> ... l (h d)')
        output = self.output_linear(output)

        return output

    def _rel_pos_attn_term(self, q, k_len, pos_enc):
        # The formula below is from the Funnel Transformer paper, page 13.
        # Q is (B, H, L, D); pos_enc is (L * 4, H * D), r_kernel is (H, H * D, D), so (pos_enc @ r_kernel) is
        # (H, L * 4, D) and that transposed is (H, D, L * 4), yielding a scores tensor (B, H, L, L * 4) which
        # we now have to cleverly manipulate into (B, H, L, L). Apparently this gives the same results as
        # a gather operation, although to be honest I don't understand why.
        scores = (q + self.r_r_bias) @ (pos_enc @ self.r_kernel).transpose(-2, -1) * q.shape[-1] ** -0.5

        temp_shape1 = list(scores.shape)
        temp_shape2 = temp_shape1.copy()

        # While this operation yields a tensor with the same dimensions as if we had simply transposed
        # the last two dimensions of `scores` (B, H, L * 4, L), it is important to understand that the
        # elements of the resulting tensor are NOT laid out in the same way as if we had simply called
        # `scores.transpose(-1, -2)`; the strides are different.
        temp_shape1[-2], temp_shape1[-1] = temp_shape1[-1], temp_shape1[-2]
        scores = scores.reshape(temp_shape1)

        # Remove 1 or 2 positions from the left end of the tensor across the long L * 4 dimension
        shift = 1 + (q.shape[-2] != k_len)
        temp_shape2[-1] -= shift
        scores = scores.narrow(-2, shift, temp_shape2[-1])

        # Now reshape to (B, H, L, (L * 4) - (1 or 2))
        scores = scores.reshape(temp_shape2)
        scores = scores.narrow(-1, 0, k_len)  # Chop off the rightmost elements to get (B, H, L, L)

        return scores

    def _update_kv_cache(self, k, v) -> Tuple[Tensor, Tensor]:
        # Register for automatic cache cleanup when we exit the cached kv context
        self.live_attention_modules.add(self)

        # We're being fed new keys and values one token at a time- self-attention case
        if k.shape[-2] == 1:
            # When we're using sparse attention, we only need to cache the keys and values that are
            # actually going to be attended to
            if self.sparse_attention:
                config = self.sparse_attention.sparsity_config
                num_blocks = config.window_size + int(config.include_cls)
                block_size = config.block
                cache_length, cache_offset = num_blocks * block_size, int(config.include_cls) * block_size
            else:
                cache_length, cache_offset = self.kv_cache_length, 0
                block_size = 0

            if self.key_cache is None:
                self.key_cache = k.new_zeros([k.shape[0], cache_length, k.shape[-1]])
                self.value_cache = v.new_zeros([v.shape[0], cache_length, v.shape[-1]])

            # We've overshot the kv cache size
            if self.sparse_attention and self.cache_index >= cache_length:
                local_index = self.cache_index % block_size
                kv_index = cache_length - block_size + local_index

                # Shift the kv cache leftward by one block, discarding the leftmost one
                if local_index == 0:
                    self.key_cache[:, cache_offset:kv_index] = self.key_cache[:, cache_offset + block_size:].clone()
                    self.value_cache[:, cache_offset:kv_index] = self.value_cache[:, cache_offset + block_size:].clone()
            else:
                kv_index = self.cache_index

            self.key_cache[:, kv_index] = k.squeeze(-2)
            self.value_cache[:, kv_index] = v.squeeze(-2)
            self.cache_index += 1

            return self.key_cache[:, :kv_index + 1], self.value_cache[:, :kv_index + 1]

        # We're being fed a bunch of keys and values all at once- cross-attention case.
        else:
            self.key_cache = k
            self.value_cache = v
            self.precomputed_kv = True
            return k, v

    # Class variables
    kv_cache_length: int = None
    live_attention_modules: Set['Attention'] = None

    @classmethod
    @contextmanager
    def kv_cache(cls, max_seq_length: int):
        cls.kv_cache_length = max_seq_length
        cls.live_attention_modules = set()

        yield

        cls.kv_cache_length = None
        for module in cls.live_attention_modules:
            module.cache_index = 0
            module.precomputed_kv = False
            module.key_cache = None
            module.value_cache = None

        cls.live_attention_modules = None

    @classmethod
    def update_kv_cache(cls, live_sample_mask: Tensor):
        for module in cls.live_attention_modules:
            module.key_cache = module.key_cache[live_sample_mask, ...]
            module.value_cache = module.value_cache[live_sample_mask, ...]


def positional_encodings_like(x: Tensor, start: int = 0):
    length = x.shape[-2]
    return get_positional_encodings(start, start + length, x.shape[-1], x.device, x.dtype)

@lru_cache(maxsize=10)
def get_positional_encodings(start: int, end: int, d_model: int, device, dtype) -> Tensor:
    position_ids = torch.arange(start, end, 1.0, dtype=dtype, device=device)
    return gather_positional_encodings(position_ids, d_model, device, dtype)

# Gather positional encodings for a tensor of positional indices
def gather_positional_encodings(positions: Tensor, d_model: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    d_model_half = d_model // 2
    frequencies = torch.arange(d_model_half, dtype=dtype, device=device)
    periods = 1 / (10000 ** (frequencies / d_model_half))

    angles = positions[:, None] * periods[None]  # noqa; Outer product
    encodings = torch.empty(positions.shape[0], d_model, dtype=dtype, device=device)
    encodings[:, ::2] = angles.sin()
    encodings[:, 1::2] = angles.cos()
    return encodings
