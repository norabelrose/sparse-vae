from dataclasses import *
from functools import lru_cache
from numpy import prod
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from typing import *
import torch
import torch.nn.functional as F


@dataclass
class AttentionState:
    hparams: AttributeDict
    input_mask: Optional[Tensor] = None       # 1 for positions we should ignore (like padding), 0 otherwise

    current_block: int = field(init=False, default=0)
    input_device: torch.device = field(init=False, default=None)
    input_dtype: torch.dtype = field(init=False, default=None)
    input_seq_len: int = field(init=False, default=None)

    # True inside a 'with attention_state.begin_block()' block
    block_begin_flag: bool = field(init=False, default=False)

    # When True, AttentionState will cache a list of all the mask tensors it computes for each block of the model.
    # These masks can then be reused, e.g. by a VAE decoder.
    cache_masks: bool = False
    upsampling: bool = False

    def __post_init__(self):
        # Wrap these functions in functools.lru_cache() objects so that we don't compute the same values multiple times
        num_scales = len(self.hparams.scaling_factors)
        cache_max_size = num_scales * 2 + 1

        self.positional_encodings_for_step = lru_cache(maxsize=cache_max_size)(self.positional_encodings_for_step)
        self.scaled_input_mask_for_block = lru_cache(maxsize=num_scales)(self.scaled_input_mask_for_block)

    def __hash__(self):  # Necessary for using functools.lru_cache on instance methods
        return id(self)

    def get_attention_mask(self) -> Tensor:
        input_mask = self.input_mask if not self.current_block else self.scaled_input_mask_for_block(self.current_block)
        return None if input_mask is None else input_mask[:, None, None, :]

    def get_positional_encodings(self) -> Union[Tensor, List[Tensor]]:
        assert self.input_seq_len is not None, \
            "configure_for_input() hasn't been called yet, so I don't know how big to make the positional encodings."

        return self.positional_encodings_for_step(self.current_block, self.block_begin_flag, self.input_seq_len,
                                                  self.input_dtype, self.input_device)

    # This method should be called before any other AttentionState methods are called in a forward pass
    def configure_for_input(self, x: Dict[str, Any]):
        # Save information about the input for calls to, i.e., get_positional_encoding()
        data = x['input']
        self.input_seq_len, self.input_dtype, self.input_device = data.shape[1], data.dtype, data.device
        self.input_mask = x.get('input_mask')

    # Mask which is 0 where a position is [CLS], 1 otherwise
    def get_not_cls_mask(self) -> Optional[Tensor]:
        if not self.hparams.separate_cls:
            return None

        q_seq_len = k_seq_len = self.input_seq_len
        q_scale, k_scale = self.query_and_key_strides_for_step(self.current_block, self.block_begin_flag,
                                                               count_from_end=False)

        if self.upsampling:
            q_seq_len *= q_scale
            k_seq_len *= k_scale
        else:
            q_seq_len //= q_scale
            k_seq_len //= k_scale

        mask = torch.ones([q_seq_len - 1, k_seq_len - 1], dtype=self.input_dtype, device=self.input_device)
        return F.pad(mask, (1, 0, 1, 0))

    # What is the total scaling factor at block N?
    def query_and_key_strides_for_step(self, block_index: int, block_begin_flag: bool,
                                       count_from_end: bool = True) -> Tuple[int, int]:
        pooled_q_unpooled_k = block_begin_flag and self.current_block > 0

        hparams = self.hparams
        factors = hparams.scaling_factors
        if self.upsampling and count_from_end:
            factors = factors[::-1]  # Count from the last block when upsampling

        q_stride = int(prod(factors[:block_index]))
        shift = factors[block_index - 1] if pooled_q_unpooled_k else 1
        k_stride = q_stride // shift

        return q_stride, k_stride

    # This method is wrapped in a functools.lru_cache() object in __post_init__, with the maxsize set dynamically
    def scaled_input_mask_for_block(self, block_index: int):
        prev_mask = self.input_mask if block_index == 1 else self.scaled_input_mask_for_block(block_index - 1)

        # Do min pooling so that we make sure we don't include any padding tokens when we downsample
        return self._scale_tensor(prev_mask, self.hparams.scaling_factors[block_index - 1], mode="min")

    # This method should be called at the end of a forward pass
    def reset(self):
        self.current_block = 0
        self.input_mask = None
        self.scaled_input_mask_for_block.cache_clear()
        self.upsampling = False

    # Either downsample or upsample the tensor, whichever is appropriate for the current block.
    def scale_input(self, x: Tensor) -> Tensor:
        hparams = self.hparams
        factors = hparams.scaling_factors

        # Sanity check
        assert self.current_block < len(factors), \
            "We ran out of scaling factors to use. Did you forget to call AttentionState.reset()?"

        # We assume that scale_input() will only be called once at the end of each block
        scaling_factor = factors[self.current_block]
        self.current_block += 1

        return self._scale_tensor(x, scaling_factor)

    # Returns one or more Tensors of sinusoidal positional encodings whose sequence length dimension is equal to the
    # initial input sequence length (if upsampling=False) or the final output length (if upsampling=True). These
    # encodings can then be downsampled by get_positional_encodings_for_step() and used in FunnelBlocks with compressed
    # sequence lengths. The functools.lru_cache() wrapper automatically caches the result of this method for a given
    # set of parameters- which should be constant within a forward pass, and often across forward passes.
    @lru_cache(maxsize=1)
    def get_base_positional_encodings(self, seq_len: int, dtype: torch.dtype,
                                      device: torch.device) -> Union[Tensor, List[Tensor]]:
        hparams = self.hparams
        positional_encoding_type = hparams.positional_encoding_type

        if self.upsampling:
            seq_len *= prod(hparams.scaling_factors)  # What's the sequence length we WILL have at the end?
        
        # Values and routines used by all three positional encoding types
        d_model = hparams.d_model
        d_model_half = d_model // 2
        frequencies = torch.arange(d_model_half, dtype=dtype, device=device)
        periods = 1 / (10000 ** (frequencies / d_model_half))

        def get_sinusoidal_encodings(start: int, stop: int):
            position_ids = torch.arange(start, stop, 1.0, dtype=dtype, device=device)
            angles = position_ids[:, None] * periods[None]  # Outer product
            return torch.sin(angles), torch.cos(angles)

        # Relative shift method of relative positional encoding- only used with softmax attention
        if positional_encoding_type == 'rel_shift':
            # Create sinusoidal encodings for all posible offsets (positive and negative) between tokens
            sines, cosines = get_sinusoidal_encodings(-seq_len * 2, seq_len * 2)
            return torch.cat([sines, cosines], dim=-1)
        else:
            # Factorized relative positional encodings are just absolute encodings with extra steps
            sines, cosines = get_sinusoidal_encodings(0, seq_len)

            if positional_encoding_type == 'absolute':
                return torch.cat([sines, cosines], dim=-1)

            elif positional_encoding_type == 'factorized':
                query_enc1 = torch.cat([sines, sines], dim=-1)
                query_enc2 = torch.cat([cosines, sines], dim=-1)
                key_enc1 = torch.cat([cosines, cosines], dim=-1)
                key_enc2 = torch.cat([-sines, cosines], dim=-1)

                return [query_enc1, query_enc2, key_enc1, key_enc2]

    # Given a block index and a flag indicating whether the queries have a larger stride than the keys, return a Tensor
    # or a list of Tensors containing positional encodings appropriate for that stage of computation. At runtime this
    # function is wrapped with functools.lru_cache() with the maxsize parameter determined dynamically.
    def positional_encodings_for_step(self, block_index: int, pooled_q_unpooled_k: bool, seq_len: int,
                                      dtype: torch.dtype, device: torch.device) -> Union[Tensor, List[Tensor]]:
        # Either a Tensor or a list of Tensors
        base_encodings = self.get_base_positional_encodings(seq_len, dtype, device)
        q_stride, k_stride = self.query_and_key_strides_for_step(block_index, pooled_q_unpooled_k)

        hparams = self.hparams
        encoding_type = hparams.positional_encoding_type

        if encoding_type == 'rel_shift':
            # All possible relative positional offsets at this scale, from greatest to least
            rel_offsets = torch.arange(seq_len, 1 - seq_len, -k_stride, dtype=torch.long, device=device)

            # Gather the relative positional encodings that are relevant for this sequence length
            zero_offset = seq_len * 2
            rel_offsets = rel_offsets[:, None] + zero_offset
            rel_offsets = rel_offsets.expand(rel_offsets.size(0), hparams.d_model)
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

        if not self.upsampling:
            x = self._prepare_for_pooling(x, scaling_factor)

            stride = (scaling_factor, 1)
            x = x[:, None, :, :]

            if mode == "mean":
                x = F.avg_pool2d(x, stride, stride=stride, ceil_mode=True)
            elif mode == "min":
                x = -F.max_pool2d(-x, stride, stride=stride, ceil_mode=True)
            else:
                raise NotImplementedError
        else:
            # For now we only support nearest-neighbor interpolation
            x = F.interpolate(x, scale_factor=scaling_factor, mode='nearest')

        return x.squeeze(1)

    def _prepare_for_pooling(self, x: Tensor, scaling_factor: int):
        # Copy the [CLS] token (scaling_factor - 1) times to make sure it doesn't get pooled into the adjacent tokens
        if self.hparams.separate_cls:
            cls_token = x.narrow(-2, 0, 1)           # The [CLS] token across all batches etc.

            shift = scaling_factor - 1               # We're magnifying [CLS] by scaling_factor
            x = x.roll(shift, -2)                    # Roll to the right to make room for the bigger [CLS] token
            x.narrow(-2, 0, shift).copy_(cls_token)  # Overwrite the last few tokens with [CLS]

        return x

