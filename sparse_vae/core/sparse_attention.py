from dataclasses import dataclass
from functools import lru_cache
# from triton.ops.blocksparse import softmax
from .sparse_softmax import softmax
from .sparse_matmul import matmul
import torch


# Frozen and therefore hashable, and therefore usable with functools.lru_cache()
@dataclass(frozen=True)
class SparseAttention:
    block_size: int = 16
    causal: bool = True
    include_cls: bool = True
    num_heads: int = 8
    max_seq_len: int = 2 ** 16
    window_size: int = 4

    def __post_init__(self):
        assert self.max_seq_len % self.block_size == 0

    @lru_cache()
    def get_master_layout(self):
        num_blocks = self.max_seq_len // self.block_size
        layout = torch.zeros(num_blocks, num_blocks, dtype=torch.int64)

        num_sides = 1 if self.causal else 2
        left_context = sum(divmod(self.window_size, num_sides))  # Round up
        right_context = self.window_size - left_context

        for offset in range(left_context):
            shifted = layout[offset:]  # Remove top N rows
            shifted.fill_diagonal_(1)  # ...and fill the diagonal of this submatrix

        for offset in range(1, right_context):
            shifted = layout[:, offset:]  # Remove left N columns
            shifted.fill_diagonal_(1)  # ...and fill the diagonal of this submatrix

        # Always attend to the blocks containing the [CLS] token (the first block in the sequence)
        if self.include_cls:
            layout[:, 0] = 1

        return layout[None, ...].repeat(self.num_heads, 1, 1)

    @lru_cache(maxsize=4096)
    def get_sdd_op(self, num_blocks: int):
        layout = self.get_master_layout()[..., :num_blocks, :num_blocks]
        return matmul(layout, self.block_size, 'sdd', trans_b=True)

    @lru_cache(maxsize=4096)
    def get_softmax_op(self, num_blocks: int):
        layout = self.get_master_layout()[..., :num_blocks, :num_blocks]
        return softmax(layout, self.block_size)

    @lru_cache(maxsize=4096)
    def get_dsd_op(self, num_blocks: int):
        layout = self.get_master_layout()[..., :num_blocks, :num_blocks]
        return matmul(layout, self.block_size, 'dsd')

    def __call__(self, q, k, v, attn_mask = None, key_padding_mask = None):
        seq_len = q.shape[-2]
        assert seq_len == k.shape[-2] == v.shape[-2]    # Self-attention
        assert seq_len <= self.max_seq_len

        num_blocks = seq_len // self.block_size
        sdd, sm, dsd = self.get_sdd_op(num_blocks), self.get_softmax_op(num_blocks), self.get_dsd_op(num_blocks)
        scores = sdd(q, k)
        dist = sm(
            scores,
            attn_mask=attn_mask.half() if attn_mask is not None else None,
            key_padding_mask=key_padding_mask.half() if key_padding_mask is not None else None,
            scale=k.shape[-1] ** -0.5
        )
        return dsd(dist, v)
