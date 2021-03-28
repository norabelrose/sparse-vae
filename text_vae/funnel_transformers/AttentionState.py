from einops import rearrange
from functools import lru_cache
from numpy import cumprod
from omegaconf import DictConfig
from torch import Tensor
from typing import *
import os
import torch
import torch.nn.functional as F


class AttentionState:
    def __init__(self, hparams: DictConfig):
        super(AttentionState, self).__init__()

        # Transient state
        self.block_begin_flag = False
        self.current_block = 0
        self._upsampling = False
        self.input_device = None
        self.max_seq_len = None
        self.padding_mask = None      # 1 for positions we should ignore (like padding), 0 otherwise

        # Configuration
        self.positional_attention_type = hparams.positional_attention_type
        self.scaling_factors = list(hparams.scaling_factors)    # Copy them since we'll mutate them
        self.strides = [1] + list(cumprod(self.scaling_factors))
        self.separate_cls = hparams.separate_cls
        self.shared = False

        # Wrap these functions in functools.lru_cache() objects so that we don't compute the same values multiple times
        if not os.environ.get('DILL_COMPATIBILITY'):
            num_scales = len(self.scaling_factors) + 1
            self.padding_mask_with_length = lru_cache(maxsize=num_scales)(self.padding_mask_with_length)

    def __hash__(self):  # Necessary for using functools.lru_cache on instance methods
        return id(self)

    def get_positional_encodings(self, d_model: int, device: torch.device) -> Union[Tensor, List[Tensor]]:
        assert self.max_seq_len is not None, \
            "configure_for_input() hasn't been called yet, so I don't know how big to make the positional encodings."

        q_stride, k_stride = self.query_and_key_strides_for_step(self.current_block, self.block_begin_flag)
        return positional_encodings_for_strides(
            attn_type=self.positional_attention_type,
            q_stride=q_stride,
            k_stride=k_stride,
            seq_len=self.max_seq_len,
            d_model=d_model,
            device=device,
            separate_cls=self.separate_cls
        )

    # This method should be called before any other AttentionState methods are called in a forward pass
    def configure_for_input(self,
                            seq_len: int,
                            padding_mask: Optional[Tensor] = None,
                            segment_ids: Optional[Tensor] = None,
                            upsampling: bool = False
                            ):
        # Clean up from the previous forward pass
        self.current_block = 0
        self.upsampling = upsampling
        if hasattr(self.padding_mask_with_length, 'cache_clear'):
            self.padding_mask_with_length.cache_clear()

        self.segment_ids = segment_ids

        # Save information about the input for calls to, i.e., get_positional_encoding()
        self.max_seq_len = seq_len
        self.padding_mask = padding_mask

    @property
    def pooled_q_unpooled_k(self) -> bool:
        return not self.upsampling and self.block_begin_flag and self.current_block > 0

    def rewind(self):
        self.current_block = 0

    @property
    def upsampling(self) -> bool:
        return self._upsampling

    @upsampling.setter
    def upsampling(self, value: bool):
        if value == self._upsampling:
            return

        self.current_block = 0
        self.scaling_factors.reverse()
        self.strides.reverse()
        self._upsampling = value

    def get_segment_matrix(self) -> Optional[Tensor]:
        if self.segment_ids is None:
            return None

        q_stride, k_stride = self.query_and_key_strides_for_step(self.current_block, self.block_begin_flag)
        q_seq_len = self.max_seq_len // q_stride
        k_seq_len = self.max_seq_len // k_stride

        seg_ids_q = self.scale_tensor(self.segment_ids, q_seq_len)
        seg_ids_k = self.scale_tensor(self.segment_ids, k_seq_len)
        segment_matrix = (seg_ids_q.unsqueeze(-1) == seg_ids_k.unsqueeze(-2))

        # Treat [cls] as in the same segment as both A & B
        cls_matrix = seg_ids_q.eq(2).unsqueeze(-1) | seg_ids_k.eq(2).unsqueeze(-2)
        return segment_matrix | cls_matrix

    # What is the total scaling factor at block N?
    def query_and_key_strides_for_step(self, block_index: int, block_begin_flag: bool) -> Tuple[int, int]:
        pooled_q_unpooled_k = not self.upsampling and block_begin_flag and block_index > 0
        q_stride = self.strides[block_index]
        k_stride = self.strides[block_index - 1] if pooled_q_unpooled_k else q_stride
        return q_stride, k_stride

    # This method is wrapped in a functools.lru_cache() object in __init__, with the maxsize set dynamically
    def padding_mask_with_length(self, length: int):
        if self.padding_mask is None or length == self.padding_mask.shape[-1]:
            return self.padding_mask

        # Do min pooling so that we make sure we don't include any padding tokens when we downsample
        return self.scale_tensor(self.padding_mask, length, mode="max")

    # Used for scaling input tensors and padding masks
    def scale_tensor(self, x: Tensor, new_length: int, mode: str = "mean") -> Optional[Tensor]:
        # Don't do unnecessary work
        if x is None:
            return None
        if new_length == x.shape[1]:
            return x

        is_bool = x.dtype == torch.bool
        if is_bool:
            x = x.float()

        if x.ndim < 3:
            has_channels = False
            x = x.unsqueeze(-1)
        else:
            has_channels = True

        if not self._upsampling:
            scaling_factor = x.shape[1] // new_length
            x = _prepare_for_pooling(x, scaling_factor, self.separate_cls)
            x = rearrange(x, '... l d -> ... d l')

            if mode == "mean":
                x = F.adaptive_avg_pool1d(x, output_size=new_length)  # noqa
            elif mode == "max":
                x = F.adaptive_max_pool1d(x, output_size=new_length)
            elif mode == "min":
                x = -F.adaptive_max_pool1d(-x, output_size=new_length)
            else:
                raise NotImplementedError
        else:
            x = rearrange(x, '... l d -> ... d l')
            x = F.interpolate(x, size=new_length)

        x = rearrange(x, '... d l -> ... l d')
        if not has_channels:
            x = x.squeeze(-1)

        if is_bool:
            x = x.bool()

        return x

# Returns a Tensor or a list of Tensors containing positional encodings appropriate for the given strides. At
# runtime this function is wrapped with functools.lru_cache() with the maxsize parameter determined dynamically.
@lru_cache(maxsize=10)
def positional_encodings_for_strides(attn_type: str, q_stride: int, k_stride: int, seq_len: int, d_model: int,
                                     device: torch.device, separate_cls: bool) -> Union[Tensor, List[Tensor]]:
    # Either a Tensor or a list of Tensors
    base_encodings = get_base_positional_encodings(attn_type, 1, seq_len, d_model, device)

    if attn_type == 'rel_shift':
        # All possible relative positional offsets at this scale, from greatest to least
        rel_offsets = torch.arange(seq_len, 1 - seq_len, -k_stride, dtype=torch.long, device=device)

        # Gather the relative positional encodings that are relevant for this sequence length
        zero_offset = seq_len * 2
        rel_offsets = rel_offsets[:, None] + zero_offset
        rel_offsets = rel_offsets.expand(rel_offsets.size(0), d_model)
        return base_encodings.gather(0, rel_offsets)
    else:
        # With absolute positional encodings, we have two encoding tensors; one for queries and one for keys
        if attn_type == 'absolute':
            q_encodings = k_encodings = [base_encodings]
        # With factorized relative encodings, we have four encoding tensors; two for queries and two for keys
        elif attn_type == 'factorized':
            q_encodings, k_encodings = base_encodings[:2], base_encodings[2:]
        else:
            raise ValueError(f"Invalid attention type '{attn_type}'")

        q_encodings = [_prepare_for_pooling(x, q_stride, separate_cls)[::q_stride] for x in q_encodings]
        k_encodings = [_prepare_for_pooling(x, k_stride, separate_cls)[::k_stride] for x in k_encodings]

        return q_encodings + k_encodings

# Returns one or more Tensors of sinusoidal positional encodings whose sequence length dimension is equal to the
# initial input sequence length (if upsampling=False) or the final output length (if upsampling=True). These
# encodings can then be downsampled by positional_encodings_for_strides() and used in blocks with compressed
# sequence lengths. The functools.lru_cache() decorator automatically caches the result of this method for a given
# set of parameters- which should be constant within a forward pass, and often across forward passes.
@lru_cache(maxsize=5)
def get_base_positional_encodings(attn_type: str, stride: int, seq_len: int, d_model: int,
                                  device: torch.device) -> Union[Tensor, List[Tensor]]:
    # Values and routines used by all three positional encoding types
    d_model_half = d_model // 2
    frequencies = torch.arange(d_model_half, dtype=torch.float32, device=device)
    periods = 1 / (10000 ** (frequencies / d_model_half))

    def get_sinusoidal_encodings(start: int, stop: int):
        position_ids = torch.arange(start, stop, stride, dtype=torch.float32, device=device)
        angles = position_ids[:, None] * periods[None]  # noqa; Outer product
        return angles.sin(), angles.cos()

    # Relative shift method of relative positional encoding- only used with softmax attention
    if attn_type == 'rel_shift':
        # Create sinusoidal encodings for all posible offsets (positive and negative) between tokens
        sines, cosines = get_sinusoidal_encodings(-seq_len * 2, seq_len * 2)
        return torch.cat([sines, cosines], dim=-1)
    else:
        # Factorized relative positional encodings are just absolute encodings with extra steps
        sines, cosines = get_sinusoidal_encodings(0, seq_len)

        if attn_type == 'absolute':
            return torch.cat([sines, cosines], dim=-1)

        elif attn_type == 'factorized':
            query_enc1 = torch.cat([sines, sines], dim=-1)
            query_enc2 = torch.cat([cosines, sines], dim=-1)
            key_enc1 = torch.cat([cosines, cosines], dim=-1)
            key_enc2 = torch.cat([-sines, cosines], dim=-1)

            return [query_enc1, query_enc2, key_enc1, key_enc2]


def _prepare_for_pooling(x: Tensor, scaling_factor: int, separate_cls: bool):
    # Copy the [CLS] token (scaling_factor - 1) times to make sure it doesn't get pooled into the adjacent tokens
    if separate_cls:
        cls_token = x.narrow(-2, 0, 1)  # The [CLS] token across all batches etc.

        shift = scaling_factor - 1  # We're magnifying [CLS] by scaling_factor
        x = x.roll(shift, -2)  # Roll to the right to make room for the bigger [CLS] token
        x.narrow(-2, 0, shift).copy_(cls_token)  # Overwrite the last few tokens with [CLS]

    return x
