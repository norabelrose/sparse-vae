from dataclasses import dataclass
from functools import lru_cache, partial
from triton.ops.blocksparse import matmul, softmax
from typing import ClassVar, List

import random
import torch


# Frozen and therefore hashable, and therefore usable with functools.lru_cache()
@dataclass(frozen=True)
class SparseAttention:
    block_size: int = 32
    causal: bool = True
    include_cls: bool = True
    num_heads: int = 8
    max_seq_len: int = 115_200
    window_size: int = 4

    _op_caches: ClassVar[dict] = {}

    def __post_init__(self):
        assert self.max_seq_len % self.block_size == 0

        step = 512 // self.block_size
        block_counts = list(range(step, self.max_seq_len // step, step))
        random.shuffle(block_counts)     # Makes the tqdm ETAs more accurate
    
    def create_luts_aot(self, block_counts: List[int]):
        from multiprocessing import Pool
        from tqdm.auto import tqdm

        print(f"Creating sparse attention lookup tables ahead of time")
        with Pool(8) as p:
            for _ in tqdm(p.imap_unordered(partial(_get_lut, self), block_counts), total=len(block_counts)):
                pass

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
    
    @lru_cache()
    def get_matmul_op(self, num_blocks: int, **kwargs):
        layout = self.get_master_layout()[..., :num_blocks, :num_blocks]
        op = matmul(layout, self.block_size, **kwargs)
        
        return op
    
    @lru_cache()
    def get_softmax_op(self, num_blocks: int):
        layout = self.get_master_layout()[..., :num_blocks, :num_blocks]
        op = softmax(layout, self.block_size)
        
        return op

    def __call__(self, q, k, v, attn_mask = None, key_padding_mask = None):
        seq_len = q.shape[-2]
        assert seq_len == k.shape[-2] == v.shape[-2]    # Self-attention
        assert seq_len <= self.max_seq_len

        num_blocks = seq_len // self.block_size
        sdd = self.get_matmul_op(num_blocks, mode='sdd', trans_b=True)
        sm = self.get_softmax_op(num_blocks)
        dsd = self.get_matmul_op(num_blocks, mode='dsd')
        scores = sdd(q, k)
        dist = sm(
            scores,
            attn_mask=attn_mask.half() if attn_mask is not None else None,
            is_causal=self.causal,
            key_padding_mask=key_padding_mask.half() if key_padding_mask is not None else None,
            scale=k.shape[-1] ** -0.5
        )
        return dsd(dist, v)

def _get_lut(attn, count: int):
    return (attn.get_matmul_op('dsd', count), attn.get_matmul_op('sdd', count), attn.get_softmax_op(count))
