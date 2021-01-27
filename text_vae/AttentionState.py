from functools import lru_cache
from numpy import cumprod
from omegaconf import OmegaConf
from torch import Tensor
from typing import *
import torch
import torch.nn.functional as F


class AttentionState:
    def __init__(self, hparams: OmegaConf):
        # Transient state
        self.block_begin_flag = False
        self.current_block = 0
        self._upsampling = False
        self.input_device = None
        self.input_dtype = None
        self.padding_mask = None      # 1 for positions we should ignore (like padding), 0 otherwise
        self.max_seq_len = None

        # Configuration
        self.d_model = hparams.d_model
        self.positional_encoding_type = hparams.positional_encoding_type
        self.scaling_factors = list(hparams.scaling_factors)    # Copy them since we'll mutate them
        self.strides = [1] + list(cumprod(self.scaling_factors))
        self.separate_cls = hparams.separate_cls
        self.shared = False

        # Wrap these functions in functools.lru_cache() objects so that we don't compute the same values multiple times
        num_scales = len(self.scaling_factors) + 1
        cache_max_size = num_scales * 2 + 1

        self.positional_encodings_for_strides = lru_cache(maxsize=cache_max_size)(self.positional_encodings_for_strides)
        self.scaled_padding_mask_for_stride = lru_cache(maxsize=num_scales)(self.scaled_padding_mask_for_stride)

    def __hash__(self):  # Necessary for using functools.lru_cache on instance methods
        return id(self)

    def get_attention_mask(self) -> Tensor:
        padding_mask = self.scaled_padding_mask_for_stride(self.strides[self.current_block])    # float
        return None if padding_mask is None else padding_mask[:, None, :, None]

    def get_padding_mask(self) -> Tensor:
        return self.scaled_padding_mask_for_stride(self.strides[self.current_block]).bool()

    def get_positional_encodings(self) -> Union[Tensor, List[Tensor]]:
        assert self.max_seq_len is not None, \
            "configure_for_input() hasn't been called yet, so I don't know how big to make the positional encodings."

        q_stride, k_stride = self.query_and_key_strides_for_step(self.current_block, self.block_begin_flag)
        return self.positional_encodings_for_strides(q_stride, k_stride, self.max_seq_len, self.input_dtype,
                                                     self.input_device)

    # This method should be called before any other AttentionState methods are called in a forward pass
    def configure_for_input(self,
                            seq_len: int,
                            dtype: torch.dtype,
                            device: torch.device,
                            padding_mask: Tensor,
                            upsampling: bool = False
                            ):
        # Clean up from the previous forward pass
        self.current_block = 0
        self.scaled_padding_mask_for_stride.cache_clear()
        self.upsampling = upsampling

        # Save information about the input for calls to, i.e., get_positional_encoding()
        self.max_seq_len, self.input_dtype, self.input_device = seq_len, dtype, device
        self.padding_mask = padding_mask

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

    # Mask which is 0 where a position is [CLS], 1 otherwise
    def get_not_cls_mask(self) -> Optional[Tensor]:
        if not self.separate_cls:
            return None

        q_stride, k_stride = self.query_and_key_strides_for_step(self.current_block, self.block_begin_flag)

        q_seq_len = self.max_seq_len // q_stride
        k_seq_len = self.max_seq_len // k_stride

        mask = torch.ones([q_seq_len - 1, k_seq_len - 1], dtype=self.input_dtype, device=self.input_device)
        return F.pad(mask, (1, 0, 1, 0))

    # What is the total scaling factor at block N?
    def query_and_key_strides_for_step(self, block_index: int, block_begin_flag: bool) -> Tuple[int, int]:
        pooled_q_unpooled_k = not self.upsampling and block_begin_flag and block_index > 0
        q_stride = self.strides[block_index]
        k_stride = self.strides[block_index - 1] if pooled_q_unpooled_k else q_stride
        return q_stride, k_stride

    # This method is wrapped in a functools.lru_cache() object in __init__, with the maxsize set dynamically
    def scaled_padding_mask_for_stride(self, stride: int):
        if stride == 1 or self.padding_mask is None:
            return self.padding_mask

        ascending_strides = sorted(self.strides)
        stride_index = ascending_strides.index(stride)
        prev_stride = ascending_strides[stride_index - 1]
        prev_mask = self.scaled_padding_mask_for_stride(prev_stride)

        # Do min pooling so that we make sure we don't include any padding tokens when we downsample
        return self._scale_tensor(prev_mask, stride // prev_stride, mode="min")

    # Either downsample or upsample the tensor, whichever is appropriate for the current block.
    def maybe_scale_input(self, x: Tensor) -> Tensor:
        # Sanity check
        num_factors = len(self.scaling_factors)
        assert self.current_block < num_factors + 1, \
            "We ran out of scaling factors to use. Did you forget to call AttentionState.reset()?"

        # We're at the end of the Transformer, we don't scale at all at the end
        if self.current_block == num_factors:
            return x

        # We assume that maybe_scale_input() will only be called once at the end of each block
        scaling_factor = self.scaling_factors[self.current_block]
        self.current_block += 1

        return self._scale_tensor(x, scaling_factor)

    # Returns one or more Tensors of sinusoidal positional encodings whose sequence length dimension is equal to the
    # initial input sequence length (if _upsampling=False) or the final output length (if _upsampling=True). These
    # encodings can then be downsampled by get_positional_encodings_for_strides() and used in blocks with compressed
    # sequence lengths. The functools.lru_cache() wrapper automatically caches the result of this method for a given
    # set of parameters- which should be constant within a forward pass, and often across forward passes.
    @lru_cache(maxsize=1)
    def get_base_positional_encodings(self, seq_len: int, dtype: torch.dtype,
                                      device: torch.device) -> Union[Tensor, List[Tensor]]:
        # Values and routines used by all three positional encoding types
        d_model_half = self.d_model // 2
        frequencies = torch.arange(d_model_half, dtype=dtype, device=device)
        periods = 1 / (10000 ** (frequencies / d_model_half))

        def get_sinusoidal_encodings(start: int, stop: int):
            position_ids = torch.arange(start, stop, 1.0, dtype=dtype, device=device)
            angles = position_ids[:, None] * periods[None]  # Outer product
            return torch.sin(angles), torch.cos(angles)

        # Relative shift method of relative positional encoding- only used with softmax attention
        if self.positional_encoding_type == 'rel_shift':
            # Create sinusoidal encodings for all posible offsets (positive and negative) between tokens
            sines, cosines = get_sinusoidal_encodings(-seq_len * 2, seq_len * 2)
            return torch.cat([sines, cosines], dim=-1)
        else:
            # Factorized relative positional encodings are just absolute encodings with extra steps
            sines, cosines = get_sinusoidal_encodings(0, seq_len)

            if self.positional_encoding_type == 'absolute':
                return torch.cat([sines, cosines], dim=-1)

            elif self.positional_encoding_type == 'factorized':
                query_enc1 = torch.cat([sines, sines], dim=-1)
                query_enc2 = torch.cat([cosines, sines], dim=-1)
                key_enc1 = torch.cat([cosines, cosines], dim=-1)
                key_enc2 = torch.cat([-sines, cosines], dim=-1)

                return [query_enc1, query_enc2, key_enc1, key_enc2]

    # Returns a Tensor or a list of Tensors containing positional encodings appropriate for the given strides. At
    # runtime this function is wrapped with functools.lru_cache() with the maxsize parameter determined dynamically.
    def positional_encodings_for_strides(self, q_stride: int, k_stride: int, seq_len: int,
                                         dtype: torch.dtype, device: torch.device) -> Union[Tensor, List[Tensor]]:
        # Either a Tensor or a list of Tensors
        base_encodings = self.get_base_positional_encodings(seq_len, dtype, device)
        encoding_type = self.positional_encoding_type

        if encoding_type == 'rel_shift':
            # All possible relative positional offsets at this scale, from greatest to least
            rel_offsets = torch.arange(seq_len, 1 - seq_len, -k_stride, dtype=torch.long, device=device)

            # Gather the relative positional encodings that are relevant for this sequence length
            zero_offset = seq_len * 2
            rel_offsets = rel_offsets[:, None] + zero_offset
            rel_offsets = rel_offsets.expand(rel_offsets.size(0), self.d_model)
            return base_encodings.gather(0, rel_offsets)
        else:
            # With absolute positional encodings, we have two encoding tensors; one for queries and one for keys
            if encoding_type == 'absolute':
                q_encodings = k_encodings = base_encodings
            # With factorized relative encodings, we have four encoding tensors; two for queries and two for keys
            elif encoding_type == 'factorized':
                q_encodings, k_encodings = base_encodings[:2], base_encodings[2:]
            else:
                raise ValueError(f"Invalid attention type '{encoding_type}'")

            q_encodings = [self._prepare_for_pooling(x, q_stride)[::q_stride] for x in q_encodings]
            k_encodings = [self._prepare_for_pooling(x, k_stride)[::k_stride] for x in k_encodings]

            return q_encodings + k_encodings

    # Used for scaling input tensors and padding masks
    def _scale_tensor(self, x: Tensor, scaling_factor: int, mode: str = "mean") -> Optional[Tensor]:
        # Don't do unnecessary work
        if x is None:
            return None
        if scaling_factor == 1:
            return x

        if not self._upsampling:
            x = self._prepare_for_pooling(x, scaling_factor)

            x.unsqueeze_(1)

            if mode == "mean":
                stride = (scaling_factor, 1)
                x = F.avg_pool2d(x, stride, stride=stride, ceil_mode=True)
            elif mode == "min":
                x = -F.max_pool1d(-x, kernel_size=scaling_factor, ceil_mode=True)
            else:
                raise NotImplementedError
        else:
            x = x.repeat_interleave(scaling_factor, dim=-2)

        return x.squeeze(1)

    def _prepare_for_pooling(self, x: Tensor, scaling_factor: int):
        # Copy the [CLS] token (scaling_factor - 1) times to make sure it doesn't get pooled into the adjacent tokens
        if self.separate_cls:
            cls_token = x.narrow(-2, 0, 1)           # The [CLS] token across all batches etc.

            shift = scaling_factor - 1               # We're magnifying [CLS] by scaling_factor
            x = x.roll(shift, -2).clone()            # Roll to the right to make room for the bigger [CLS] token
            x.narrow(-2, 0, shift).copy_(cls_token)  # Overwrite the last few tokens with [CLS]

        return x

