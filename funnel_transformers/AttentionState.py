from __future__ import annotations

from copy import copy
from dataclasses import *
import torch.nn.functional as F
from ..Utilities import *
if TYPE_CHECKING:   # Avoid circular dependency
    from .FunnelTransformer import FunnelConfig

# A tensor if attention_type == "rel_shift", but a list of tensors if attention_type == "factorized"
PositionalEncoding = NewType('PositionalEncoding', Union[Tensor, List[Tensor]])


@dataclass
class AttentionState:
    funnel_config: FunnelConfig
    segment_mask: Tensor = None     # 1 where Q and K are in same segment (e.g. Next Sentence Prediction), 0 otherwise
    input_mask: Tensor = None       # 1 for positions we should ignore (like padding), 0 otherwise
    not_cls_mask: Tensor = None     # 0 where position is the [CLS] token, 1 otherwise

    # True inside a 'with attention_state.pooled_q_unpooled_k()' block
    pooled_q_unpooled_k_flag: bool = field(init=False, default=False)

    # When True, AttentionState will cache a list of all the mask tensors it computes for each block of the model.
    # These masks can then be reused, e.g. by a VAE decoder.
    cache_masks: bool = True

    # Private variables
    _current_block: int = field(init=False, default=0)
    _last_seq_len: int = field(init=False, default=None)
    _last_dtype: DataType = field(init=False, default=None)
    _last_device: Device = field(init=False, default=None)
    _mask_stack: List[Tuple[Tensor, Tensor, Tensor]] = field(init=False, default_factory=list)
    _pos_encodings: List[List[PositionalEncoding]] = field(init=False, default_factory=list)

    @property
    def attention_mask(self) -> Tensor:
        return None if self.input_mask is None else self.input_mask[:, None, None, :]

    @property
    def positional_encoding(self) -> PositionalEncoding:
        if not self._pos_encodings:
            assert self._last_seq_len is not None, \
                "I haven't seen an input tensor yet, so I don't know how big to make the positional encodings."

            self._generate_pos_encodings(self._last_seq_len, self._last_dtype, self._last_device)

        # The pooled Q, unpooled K encodings are at index 1
        return self._pos_encodings[self._current_block][int(self.pooled_q_unpooled_k_flag)]

    # This method should be called before .positional_encoding is queried in a forward pass
    def configure_for_input(self, x: Tensor, input_mask: Tensor = None, seg_id: Tensor = None):
        # Only spend the FLOPs on computing new positional encodings if we actually have to (we usually shouldn't)
        new_seq_len, new_device, new_dtype = x.shape[1], x.device, x.dtype
        if (new_seq_len, new_device, new_dtype) != (self._last_seq_len, self._last_device, self._last_dtype):
            self._pos_encodings = self._generate_pos_encodings(new_seq_len, new_device, new_dtype)

            self._last_seq_len = x.shape[1]
            self._last_device = x.device
            self._last_dtype = x.dtype

        # When we reset last time, keep_masks was set to True. Let's reuse the old masks.
        if len(self._mask_stack) > 0:
            self.input_mask, self.not_cls_mask, self.segment_mask = self._mask_stack.pop()
            return

        # By default, mask out all the padding tokens
        pad_id = self.funnel_config.pad_id
        if input_mask is None and pad_id is not None:
            input_mask = (x == pad_id).float()

        self.input_mask = input_mask

        if self.funnel_config.separate_cls:
            # Mask which is 0 where a position is [CLS], 1 otherwise
            mask = torch.ones([new_seq_len - 1, new_seq_len - 1], dtype=new_dtype, device=new_device)
            self.not_cls_mask = F.pad(mask, (1, 0, 1, 0))

        # Compute the segment matrix. It's 1 where Q and K are in the same segment, 0 where they
        # are in different segments. This is used for the Next Sentence Prediction pretraining task.
        if seg_id is not None:
            # [CLS] has a 'third' segment of its own- seg_id_cls, which is 2 by default
            mask = torch.eq(seg_id, self.funnel_config.seg_id_cls)          # 1 where position is [CLS], 0 otherwise
            cls_mat = torch.unsqueeze(mask, -1) | torch.unsqueeze(mask, -2)

            seg_mat = torch.eq(torch.unsqueeze(seg_id, -1), torch.unsqueeze(seg_id, -2))
            self.segment_mask = cls_mat | seg_mat     # Treat [CLS] as in the same segment as both A & B

        if self.cache_masks:
            self._mask_stack.append((self.input_mask, self.not_cls_mask, self.segment_mask))

    def invert_block_order(self):
        self._pos_encodings.reverse()

    # This method should be called at the end of a forward pass
    def reset(self, keep_masks: bool = False):
        self._current_block = 0
        self.input_mask = None
        self.not_cls_mask = None
        self.segment_mask = None

        if not keep_masks:
            self._mask_stack.clear()

    # Either pool or upsample the tensor, which ever is appropriate for the current block.
    def scale_input(self, x: Tensor) -> Tensor:
        config = self.funnel_config
        factors = config.scaling_factors

        # Sanity check
        assert self._current_block < len(factors), \
            "We ran out of scaling factors to use. Did you forget to call AttentionState.reset()?"

        scaling_factor = factors[self._current_block]
        if scaling_factor != 1:
            x = self._scale_tensor(x, scaling_factor, config.upsampling)

            if config.upsampling:
                # We have cached masks from a previous downsampling operation, so use them
                if len(self._mask_stack) > 0:
                    self.input_mask, self.not_cls_mask, self.segment_mask = self._mask_stack.pop()

                # We don't have cached masks, so just nearest-neighbor upsample our current ones (not ideal)
                else:
                    self.input_mask = self._scale_tensor(self.input_mask, scaling_factor, True)
                    self.not_cls_mask = self._scale_tensor(self.not_cls_mask, scaling_factor, True)
                    self.segment_mask = self._scale_tensor(self.segment_mask, scaling_factor, True)
            else:
                self.input_mask = self._scale_tensor(self.input_mask, scaling_factor, config.upsampling, mode="min")

                if self.cache_masks:
                    self._mask_stack.append((self.input_mask, self.not_cls_mask, self.segment_mask))

        # Increment our _current_block counter
        self._current_block += 1
        return x

    # We use with-statements to keep track of whether we need to yield tensors that are appropriate
    # for when q.shape[1] < k.shape[1]; that is, right after a downsampling operation.
    def pooled_q_unpooled_k(self):  # with attention_state.pooled_q_unpooled_k(): ...
        return self

    # We're entering a pooled_q_unpooled_k block
    def __enter__(self):
        if self._current_block == 0:  # We don't actually do pooling before the first block
            return

        self.pooled_q_unpooled_k_flag = True
        scaling_factor = self.funnel_config.scaling_factors[self._current_block]

        # Downsample the Q portion of these masks
        self.not_cls_mask = self._stride_downsample(self.not_cls_mask, -2, scaling_factor)
        self.segment_mask = self._stride_downsample(self.segment_mask, -2, scaling_factor)

        # We don't cache pooled Q, unpooled K masks because they aren't needed for upsampling

    # We're exiting a pooled_q_unpooled_k block
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._current_block == 0:  # We don't actually do pooling before the first block
            return

        self.pooled_q_unpooled_k_flag = False
        scaling_factor = self.funnel_config.scaling_factors[self._current_block]

        # Downsample the K portion of these masks
        self.not_cls_mask = self._stride_downsample(self.not_cls_mask, -1, scaling_factor)
        self.segment_mask = self._stride_downsample(self.segment_mask, -1, scaling_factor)

        if self.cache_masks:
            self._mask_stack.append((self.input_mask, self.not_cls_mask, self.segment_mask))

    # Used for scaling multiple different kinds of tensors (pos encodings, text representations, etc.)
    def _scale_tensor(self, x: Tensor, scaling_factor: int, upsample: bool, mode: str = None) -> Tensor:
        if x is None:
            return None

        config = self.funnel_config
        mode = mode or config.pooling_type

        # Remove the [CLS] token if we're protecting it from getting scaled
        if config.separate_cls:
            if config.truncate_seq:
                x = torch.cat([x[:, :1], x[:, :-1]], dim=1)
            else:
                x = torch.cat([x[:, :1], x], dim=1)

        if not upsample:
            stride = (scaling_factor, 1)

            # ndims == 2 is for pos encodings (which have no batch dim), ndims == 3 is for text representations
            ndims = x.dim()
            assert ndims == 2 or ndims == 3 or ndims == 4

            if ndims == 2:
                x = x[:, None, :, None]
            elif ndims == 3:
                x = x[:, None, :, :]

            if mode == "mean":
                x = F.avg_pool2d(x, stride, stride=stride, ceil_mode=True)
            elif mode == "max":
                x = F.max_pool2d(x, stride, stride=stride, ceil_mode=True)
            elif mode == "min":
                x = -F.max_pool2d(-x, stride, stride=stride, ceil_mode=True)
            else:
                raise NotImplementedError
            if ndims == 2:
                x = x.squeeze(-1).squeeze(1)
            elif ndims == 3:
                x = x.squeeze(1)
        else:
            # For now we only support nearest-neighbor interpolation
            x = F.interpolate(x, scale_factor=scaling_factor, mode='nearest')

        return x

    # Downsample by stride slicing the tensor along the given axis.
    def _stride_downsample(self, x: T, axis: int, scaling_factor: int) -> T:
        if x is None:
            return None

        config = self.funnel_config
        stop = -1 if config.separate_cls and config.truncate_seq else None
        strided = slice_tensors(x, axis, stop=stop, step=scaling_factor)

        if config.separate_cls:
            cls = slice_tensors(x, axis, stop=1)
            strided = [torch.cat([x, y], axis=axis) for x, y in zip(cls, strided)]

        return strided

    # We generate and cache all the positional encoding tensors for all blocks up front. This is mainly because
    # the relative shift encodings for block N can't be directly generated by stride downsampling the encodings
    # for block N - 1. This actually *is* true for the factorized encodings, and the original implementation
    # generated those just-in-time while caching rel shift encodings ahead of time. This made the code
    # unnecessarily complicated, however, so we just cache both types of encodings.
    def _generate_pos_encodings(self, seq_len: int, dtype: DataType, device: Device):
        config = self.funnel_config
        attn_type = config.attention_type
        self._pos_encodings.clear()

        scaling_factors = copy(config.scaling_factors)
        if config.upsampling:
            seq_len *= product(scaling_factors)  # What's the sequence length we WILL have at the end?
            scaling_factors.reverse()  # Create the encodings backwards from the end

        if attn_type == "factorized":
            pos_seq = torch.arange(0, seq_len, 1.0, dtype=dtype, device=device)
            pos_seq_q, pos_seq_k = pos_seq, pos_seq
            d_model = self.net_config.d_model
            d_model_half = d_model // 2
            freq_seq = torch.arange(0, d_model_half, 1.0, dtype=dtype, device=device)
            inv_freq = 1 / (10000 ** (freq_seq / d_model_half))
            sinusoid_q = torch.einsum("...i,d->...id", pos_seq_q, inv_freq)
            sinusoid_k = torch.einsum("...i,d->...id", pos_seq_k, inv_freq)
            sin_enc_q = torch.sin(sinusoid_q)
            cos_enc_q = torch.cos(sinusoid_q)
            sin_enc_k = torch.sin(sinusoid_k)
            cos_enc_k = torch.cos(sinusoid_k)
            enc_q_1 = torch.cat([sin_enc_q, sin_enc_q], dim=-1)
            enc_k_1 = torch.cat([cos_enc_k, sin_enc_k], dim=-1)
            enc_q_2 = torch.cat([cos_enc_q, cos_enc_q], dim=-1)
            enc_k_2 = torch.cat([-sin_enc_k, cos_enc_k], dim=-1)

            encoding_block = [[enc_q_1, enc_q_2, enc_k_1, enc_k_2]]  # First block only needs one pos encoding
            self._pos_encodings.append(encoding_block)

            # Now create all the scaled down ones
            for scaling_factor in scaling_factors:
                if scaling_factor == 1:  # Ignore the placeholder 1 at the end of the list
                    break

                last_encoding = encoding_block[-1]

                q_pooled = self._stride_downsample(last_encoding[:2], 1, scaling_factor)
                k_pooled = self._stride_downsample(last_encoding[2:], 1, scaling_factor)

                encoding_block = [q_pooled + last_encoding[2:], q_pooled + k_pooled]
                self._pos_encodings.append(encoding_block)

            return [[enc_q_1, enc_q_2, enc_k_1, enc_k_2]]

        elif attn_type == "rel_shift":
            d_model = config.d_model
            d_model_half = d_model // 2
            freq_seq = torch.arange(0, d_model_half, 1.0, dtype=dtype, device=device)
            inv_freq = 1 / (10000 ** (freq_seq / d_model_half))

            # initialize an extra long position sequnece
            rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0, dtype=dtype, device=device)
            zero_offset = seq_len * 2

            sinusoid = torch.einsum("...i,d->...id", rel_pos_id, inv_freq)
            sin_enc = torch.sin(sinusoid)
            cos_enc = torch.cos(sinusoid)
            pos_enc = torch.cat([sin_enc, cos_enc], dim=-1)

            # Pre-compute and cache the rel_pos_id for all blocks
            pos_id = torch.arange(0, seq_len, dtype=dtype, device=device)

            #### Attn(pooled-q, pooled-kv) type
            pos_enc_1 = self._pos_ids_to_encoding(q_ids=pos_id, q_stride=1, k_ids=pos_id, k_stride=1,
                                                  gather_source=pos_enc, zero_offset=zero_offset, d_model=d_model)
            self._pos_encodings.append([pos_enc_1])

            # Now create all the scaled down ones
            for scaling_factor in scaling_factors:
                if scaling_factor == 1:  # Ignore the placeholder 1 at the end of the list
                    break

                #### pos_enc_1 == Attn(pooled-q, pooled-kv) and pos_enc_2 == Attn(pooled-q, unpooled-kv)
                pos_enc_1 = self._pos_ids_to_encoding(q_ids=pos_id, q_stride=scaling_factor, k_ids=pos_id,
                                                      k_stride=1, gather_source=pos_enc, zero_offset=zero_offset,
                                                      d_model=d_model)
                pos_enc_2 = self._pos_ids_to_encoding(q_ids=pos_id, q_stride=scaling_factor, k_ids=pos_id,
                                                      k_stride=scaling_factor, gather_source=pos_enc,
                                                      zero_offset=zero_offset, d_model=d_model)
                self._pos_encodings.append([pos_enc_1, pos_enc_2])

        # Now flip the encodings into the right order
        if config.upsampling:
            self._pos_encodings.reverse()

    # Only used for relative shift attention
    @staticmethod
    def _pos_ids_to_encoding(q_ids: Tensor, q_stride: int, k_ids: Tensor, k_stride: int, gather_source: Tensor,
                             zero_offset: int, d_model: int) -> Tensor:
        shift = q_stride // k_stride

        ref_point = q_ids[0] - k_ids[0]
        num_remove = shift * len(q_ids)
        max_dist = ref_point + num_remove * k_stride
        min_dist = q_ids[0] - k_ids[-1]

        rel_ids = torch.arange(max_dist, min_dist - 1, -k_stride, dtype=torch.long, device=q_ids.device)

        # gather relative positional encoding
        rel_ids = rel_ids[:, None] + zero_offset
        rel_ids = rel_ids.expand(rel_ids.size(0), d_model)
        return torch.gather(gather_source, 0, rel_ids)
