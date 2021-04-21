from deepspeed.ops.sparse_attention import SparsityConfig
from torch import Tensor
import torch


# Causal sliding window sparse attention. Unfortunately, since this is implemented with
# *block* sparse matrices, you still need to apply a causal attention mask in order to
# make sure all the right-context is hidden.
class SlidingWindowSparsityConfig(SparsityConfig):
    def __init__(self, num_heads: int, block: int = 16, window_size: int = 4, causal: bool = True,
                 include_cls: bool = True):
        super().__init__(num_heads, block, different_layout_per_head=False)
        self.causal = causal
        self.include_cls = include_cls
        self.window_size = window_size

    def make_layout(self, seq_len: int) -> Tensor:
        assert seq_len % self.block == 0
        num_blocks = seq_len // self.block

        layout = torch.zeros(num_blocks, num_blocks, dtype=torch.int64)

        num_sides = 1 if self.causal else 2
        left_context = sum(divmod(self.window_size, num_sides))  # Round up
        right_context = self.window_size - left_context

        for offset in range(left_context):
            shifted = layout[offset:]       # Remove top N rows
            shifted.fill_diagonal_(1)       # ...and fill the diagonal of this submatrix

        for offset in range(1, right_context):
            shifted = layout[:, offset:]    # Remove left N columns
            shifted.fill_diagonal_(1)       # ...and fill the diagonal of this submatrix

        # Always attend to the blocks containing the [CLS] token (the first block in the sequence)
        if self.include_cls:
            layout[:, 0] = 1

        return layout[None, ...].expand(self.num_heads, *layout.shape)
