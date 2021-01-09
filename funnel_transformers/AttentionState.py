from contextlib import contextmanager
from copy import copy
from dataclasses import *
from numpy import prod
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from typing import *
import torch
import torch.nn.functional as F

# A tensor if attention_type == "rel_shift", but a list of tensors if attention_type == "factorized"
PositionalEncoding = NewType('PositionalEncoding', Union[Tensor, List[Tensor]])


@dataclass
class AttentionState:
    hparams: AttributeDict
    segment_mask: Optional[Tensor] = None     # 1 where Q and K are in same segment, 0 otherwise
    input_mask: Optional[Tensor] = None       # 1 for positions we should ignore (like padding), 0 otherwise
    not_cls_mask: Optional[Tensor] = None     # 0 where position is the [CLS] token, 1 otherwise

    # True inside a 'with attention_state.begin_block()' block
    block_begin_flag: bool = field(init=False, default=False)

    # When True, AttentionState will cache a list of all the mask tensors it computes for each block of the model.
    # These masks can then be reused, e.g. by a VAE decoder.
    cache_masks: bool = False
    has_decoder_block: bool = False

    # Private variables
    _current_block: int = field(init=False, default=0)
    _last_seq_len: int = field(init=False, default=None)
    _last_dtype: torch.dtype = field(init=False, default=None)
    _last_device: torch.device = field(init=False, default=None)
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
        return self._pos_encodings[self._current_block][int(self.block_begin_flag)]

    # This method should be called before any AttentionState properties are queried in a forward pass
    def configure_for_input(self, x: Tensor, input_mask: Tensor = None, seg_id: Tensor = None):
        # Only spend the FLOPs on computing new positional encodings if we actually have to (we usually shouldn't)
        new_seq_len, new_device, new_dtype = x.shape[1], x.device, x.dtype
        if (new_seq_len, new_device, new_dtype) != (self._last_seq_len, self._last_device, self._last_dtype):
            self._generate_pos_encodings(new_seq_len, new_dtype, new_device)

            self._last_seq_len = x.shape[1]
            self._last_device = x.device
            self._last_dtype = x.dtype

        self.input_mask = input_mask

        if self.hparams.separate_cls:
            # Mask which is 0 where a position is [CLS], 1 otherwise
            mask = torch.ones([new_seq_len - 1, new_seq_len - 1], dtype=new_dtype, device=new_device)
            self.not_cls_mask = F.pad(mask, (1, 0, 1, 0))

        # Compute the segment matrix. It's 1 where Q and K are in the same segment, 0 where they
        # are in different segments. This is used for the Next Sentence Prediction training task.
        if seg_id is not None:
            # [CLS] has a 'third' segment of its own- seg_id_cls, which is 2 by default
            mask = seg_id.eq(self.hparams.seg_id_cls)          # 1 where position is [CLS], 0 otherwise
            cls_mat = mask.unsqueeze(-1) | mask.unsqueeze(-2)

            seg_mat = seg_id.unsqueeze(-1).eq(seg_id.unsqueeze(-2))
            self.segment_mask = cls_mat | seg_mat     # Treat [CLS] as in the same segment as both A & B

    def invert_block_order(self):
        self._pos_encodings.reverse()
    
    # We use with-statements to keep track of whether we need to yield tensors that are appropriate
    # for when q.shape[1] < k.shape[1]; that is, right after a downsampling operation.
    @contextmanager
    def begin_block(self):  # with attention_state.begin_block(): ...
        # We're entering the first layer of a new funnel block
        self._get_scaled_masks('q')
        
        yield   # Compute attention and stuff...
        
        # We're exiting the first layer of the new block
        self._get_scaled_masks('k')

    # Called twice every time we transition to a new block
    def _get_scaled_masks(self, component: str):
        dim = -1 if component == 'q' else -2

        # If we're upsampling, fetch cached masks from an earlier downsampling operation. Otherwise, average/min pool.
        if self.hparams.upsampling:
            # We must have cached masks from a previous downsampling operation— use them
            assert self._mask_stack, "We need to have cached masks from an earlier downsampling operation"
            self.input_mask, self.not_cls_mask, self.segment_mask = self._mask_stack.pop()

        else:
            if self.cache_masks:
                self._mask_stack.append((self.input_mask, self.not_cls_mask, self.segment_mask))

            if self._current_block > 0:
                self.block_begin_flag = True
                scaling_factor = self.hparams.scaling_factors[self._current_block - 1]

                # Downsample either the Q or K portion of these masks
                self.not_cls_mask = self._stride_downsample(self.not_cls_mask, dim, scaling_factor)
                self.segment_mask = self._stride_downsample(self.segment_mask, dim, scaling_factor)
                if dim == -1:
                    self.input_mask = self._stride_downsample(self.input_mask, dim, scaling_factor)

    # This method should be called at the end of a forward pass
    def reset(self):
        self._current_block = 0
        self.input_mask = None
        self.not_cls_mask = None
        self.segment_mask = None

        if self.cache_masks:
            self._mask_stack.clear()

    # Either downsample or upsample the tensor, whichever is appropriate for the current block.
    def scale_input(self, x: Tensor) -> Tensor:
        hparams = self.hparams
        factors = hparams.scaling_factors

        # Suppress the sanity check if we have a decoder block since FunnelWithDecoder will do the upsample for us
        if not self.has_decoder_block:
            # Sanity check
            assert self._current_block < len(factors), \
                "We ran out of scaling factors to use. Did you forget to call AttentionState.reset()?"

        # We assume that scale_input() will only be called once at the end of each block
        self._current_block += 1

        # Don't actually do anything if we don't have to
        scaling_factor = factors[self._current_block]
        return self._scale_tensor(x, scaling_factor, hparams.upsampling) if scaling_factor != 1 else x

    # Used for scaling multiple different kinds of tensors (pos encodings, text representations, etc.)
    def _scale_tensor(self, x: Tensor, scaling_factor: int, upsample: bool, mode: str = None) -> Optional[Tensor]:
        if x is None:
            return None

        config = self.hparams
        mode = mode or config.pooling_type

        # Remove [CLS] temporarily to protect it from the scaling
        if config.separate_cls:
            cls_token = x[:, :1]  # Save for later
            
            # If truncate_seq == True, we also remove the token at the right end of the sequence to 'make it even'
            x = x[:, 1:-1] if config.truncate_seq else x[:, 1:]

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
        
        if config.separate_cls:
            x = torch.cat([cls_token, x], dim=1)  # Tack [CLS] back on

        return x

    # Downsample by stride slicing the tensor along the given axis.
    T = TypeVar('T', bound=Union[List[Tensor], Tensor])
    def _stride_downsample(self, x: T, axis: int, scaling_factor: int) -> Optional[T]:
        if x is None:
            return None

        def slice_tensors(x, axis: int, start: int = None, stop: int = None, step: int = None):
            rank = x.dim() if torch.is_tensor(x) else x[0].dim()
            assert axis < rank
            axis = axis % rank  # for negative indices

            indices = []
            for i in range(rank):
                if i == axis:
                    indices.append(slice(start, stop, step))
                    break
                else:
                    indices.append(slice(None))  # Include all elements along this axis; equivalent to : in x[:, 5]

            return x[indices] if torch.is_tensor(x) else [tensor[indices] for tensor in x]

        config = self.hparams
        stop = -scaling_factor if config.separate_cls and config.truncate_seq else None
        strided = slice_tensors(x, axis, stop=stop, step=scaling_factor)

        if config.separate_cls:
            cls_token = slice_tensors(x, axis, stop=1)
            if torch.is_tensor(x):
                strided = torch.cat([cls_token, strided], dim=axis)
            else:
                strided = [torch.cat([x, y], dim=axis) for x, y in zip(cls_token, strided)]

        return strided

    # We generate and cache all the positional encoding tensors for all blocks up front. This is mainly because
    # the relative shift encodings for block N can't be directly generated by stride downsampling the encodings
    # for block N - 1. This actually *is* true for the factorized encodings, and the original implementation
    # generated those just-in-time while caching rel shift encodings ahead of time. This made the code
    # unnecessarily complicated, however, so we just cache both types of encodings.
    def _generate_pos_encodings(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        config = self.hparams
        attn_type = config.attention_type
        self._pos_encodings.clear()

        scaling_factors = copy(config.scaling_factors)
        if config.upsampling:
            seq_len *= prod(scaling_factors)  # What's the sequence length we WILL have at the end?
            scaling_factors.reverse()  # Create the encodings backwards from the end
        
        # Used by both factorized and rel shift
        d_model = config.d_model
        d_model_half = d_model // 2
        freq_seq = torch.arange(d_model_half, dtype=dtype, device=device)
        inv_freq = 1 / (10000 ** (freq_seq / d_model_half))

        if attn_type == "factorized":
            pos_seq = torch.arange(seq_len, dtype=dtype, device=device)
            pos_seq_q, pos_seq_k = pos_seq, pos_seq
            
            sinusoid_q = torch.einsum("i,d->id", pos_seq_q, inv_freq)  # Outer product
            sinusoid_k = torch.einsum("i,d->id", pos_seq_k, inv_freq)
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

                last_encoding = self._pos_encodings[-1][0]

                q_pooled = self._stride_downsample(last_encoding[:2], 0, scaling_factor)
                k_pooled = self._stride_downsample(last_encoding[2:], 0, scaling_factor)

                encoding_block = [q_pooled + k_pooled, q_pooled + last_encoding[2:]]
                self._pos_encodings.append(encoding_block)

            return [[enc_q_1, enc_q_2, enc_k_1, enc_k_2]]

        elif attn_type == "rel_shift":
            # initialize an extra long position sequnece
            rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0, dtype=dtype, device=device)
            zero_offset = seq_len * 2

            sinusoid = torch.einsum("i,d->id", rel_pos_id, inv_freq)  # Outer product
            sin_enc = torch.sin(sinusoid)
            cos_enc = torch.cos(sinusoid)
            pos_enc = torch.cat([sin_enc, cos_enc], dim=-1)

            # Pre-compute and cache the rel_pos_id for all blocks
            pos_id = torch.arange(seq_len, dtype=dtype, device=device)

            # Attn(pooled-q, pooled-kv) type
            pos_enc_2 = self._pos_ids_to_encoding(q_ids=pos_id, q_stride=1, k_ids=pos_id, k_stride=1,
                                                  gather_source=pos_enc, zero_offset=zero_offset, d_model=d_model)
            self._pos_encodings.append([pos_enc_2])

            # Now create all the scaled down ones
            q_stride = 1
            k_stride = 1
            for scaling_factor in scaling_factors:
                if scaling_factor == 1:  # Ignore the placeholder 1 at the end of the list
                    break
                
                q_stride *= scaling_factor
                
                # Stride pool the positional encodings
                if config.separate_cls:
                    # Under separate [cls], we treat the [cls] as the first token in
                    # the previous block of the 1st real block. Since the 1st real
                    # block always has position 1, the position of the previous block
                    # will 1 - 2**bidx, where `2 ** bidx` is the current stride.
                    cls_pos = pos_id.new_tensor([1 - q_stride])
                    if config.truncate_seq:
                        pooled_pos_id = pos_id[1:-1]
                    else:
                        pooled_pos_id = pos_id[1:]
                    pooled_pos_id = torch.cat([cls_pos, pooled_pos_id[::scaling_factor]], 0)
                else:
                    pooled_pos_id = pos_id[::scaling_factor]

                # pos_enc_1 == Attn(pooled-q, unpooled-kv) and pos_enc_2 == Attn(pooled-q, pooled-kv)
                pos_enc_2 = self._pos_ids_to_encoding(q_ids=pooled_pos_id, q_stride=q_stride, k_ids=pos_id,
                                                      k_stride=k_stride, gather_source=pos_enc,
                                                      zero_offset=zero_offset, d_model=d_model)
                pos_id = pooled_pos_id
                k_stride = q_stride
                
                pos_enc_1 = self._pos_ids_to_encoding(q_ids=pos_id, q_stride=q_stride, k_ids=pos_id,
                                                      k_stride=k_stride, gather_source=pos_enc,
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
        rel_ids = rel_ids.unsqueeze(-1) + zero_offset
        rel_ids = rel_ids.expand(rel_ids.size(0), d_model)  # Broadcast the rel positions across the embedding dim
        return torch.gather(gather_source, 0, rel_ids)
