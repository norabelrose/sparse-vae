from text_vae.core import PaddedTensor, TransformerLayer
from dataclasses import dataclass
from functools import lru_cache
from numpy import cumprod, prod
from omegaconf import OmegaConf, DictConfig
from torch import nn, Tensor
from typing import *
import torch


@dataclass
class FunnelTransformerHparams:
    block_sizes: Sequence[int] = (4, 4, 4)
    scaling_factors: Sequence[int] = (2, 4)

    d_model: int = 768
    num_heads: int = 12

    max_seq_length: Optional[int] = None
    padding_idx: Optional[int] = 0
    vocab_size: int = 30522

    # Use strided convolutions instead of average pooling for downsampling, and
    # transpose convolutions instead of nearest-neighbor upsampling
    use_convolutions: bool = True
    use_embedding: bool = True
    upsampling: bool = False  # True for the "reverse" funnel transformer; e.g. a VAE decoder

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "num_heads must divide d_model evenly"


class FunnelTransformer(nn.Module):
    def __init__(self, hparams: Union[FunnelTransformerHparams, DictConfig]):
        super().__init__()

        if isinstance(hparams, FunnelTransformerHparams):
            hparams = OmegaConf.structured(hparams)

        self.hparams = hparams

        if hparams.use_convolutions:
            conv_type = nn.ConvTranspose1d if hparams.upsampling else nn.Conv1d
            self.scalers = nn.ModuleList([
                conv_type(hparams.d_model, hparams.d_model, kernel_size=scale, stride=scale)
                for scale in hparams.scaling_factors
            ])
        else:
            self.scalers = nn.ModuleList([
                nn.Upsample(scale_factor=scale) if hparams.upsampling else nn.AvgPool1d(kernel_size=scale)
                for scale in hparams.scaling_factors
            ])

        if not hparams.upsampling and hparams.use_embedding:
            self.input_layer = nn.Sequential(
                nn.Embedding(hparams.vocab_size, hparams.d_model, padding_idx=0),
                nn.LayerNorm(hparams.d_model),
                nn.Dropout(p=0.1)
            )

        q_strides = sum(([stride] * num_layers for stride, num_layers in zip(self.strides(), hparams.block_sizes)), [])
        kv_strides = [1] + q_strides[:-1] if not hparams.upsampling else q_strides
        self.layers = nn.ModuleList([
            FunnelLayer(hparams, q_stride=q_stride, kv_stride=kv_stride)
            for q_stride, kv_stride in zip(q_strides, kv_strides)
        ])

    def strides(self) -> List[int]:
        scaling_factors = [1] + list(self.hparams.scaling_factors)
        encoder_strides = cumprod(scaling_factors).tolist()
        return encoder_strides if not self.hparams.upsampling else encoder_strides[::-1]

    # Activates cross attention for the layers specified in the list of (block index, layer index) tuples
    def configure_cross_attention(self, layers: List[Tuple[int, int]]):
        layers.sort()
        layer_iter, tuple_iter = iter(self.layers), iter(layers)

        cur_tuple = next(tuple_iter)
        for i, block_size in enumerate(self.hparams.block_sizes):
            for j in range(block_size):
                layer = next(layer_iter)

                if (i, j) == cur_tuple:
                    layer.use_cross_attention = True
                    cur_tuple = next(tuple_iter, None)
                    if not cur_tuple:
                        break

    def forward(self, x: PaddedTensor, start_block: int = 0, end_block: int = None, block_end_callback: Callable = None):
        assert isinstance(x, PaddedTensor), "The input to FunnelTransformer must be a PaddedTensor!"

        hparams = self.hparams
        if not hparams.upsampling and hparams.use_embedding:
            x = self.input_layer(x)  # x.shape == (...length, d_model)

        seq_len = x.shape[-2]
        if hparams.upsampling:
            seq_len *= prod(hparams.scaling_factors)

        blocks = self.hparams.block_sizes
        start_layer = sum(blocks[:start_block])
        layer_iter = iter(self.layers[start_layer:])

        num_blocks = len(blocks)
        start_block %= num_blocks
        end_block = end_block % num_blocks if end_block else num_blocks
        last_block = num_blocks - 1

        context = None
        if block_end_callback and start_block > 0:  # Special case; we have a callback and we're starting at a block > 0
            x, context = block_end_callback(start_block, x)

        hidden_states = []
        q = kv = x
        for block_idx, block_size in enumerate(self.hparams.block_sizes[start_block:end_block], start=start_block):
            for rel_layer_idx in range(block_size):
                layer = next(layer_iter)
                q = kv = layer(q, kv, full_seq_len=seq_len, context=context)

            hidden_states.append(kv)

            # We don't need to do any of this stuff if we just finished the last block
            if block_idx == last_block:
                break

            q = self.maybe_scale_hidden(kv, block_idx)

            # Hook for users of the module to modify the hidden states of the Transformer during a forward pass
            if block_end_callback:
                maybe_transformed_q = block_end_callback(block_idx, q)

                # Returning None from the callback means the forward pass should be aborted
                if maybe_transformed_q is None:
                    return None
                else:
                    # The callback can send us tensors to cross-attend to
                    if isinstance(maybe_transformed_q, tuple):
                        context = maybe_transformed_q[1]
                        maybe_transformed_q = maybe_transformed_q[0]
                    else:
                        context = None

                    q = kv = maybe_transformed_q

            # When we're upsampling, there's no stage where we scale Q but don't scale K and V.
            if self.hparams.upsampling:
                kv = q

        return hidden_states

    # Either downsample or upsample the tensor, whichever is appropriate for the current block.
    def maybe_scale_hidden(self, x: Tensor, cur_block: int) -> Tensor:
        # Sanity check
        num_factors = len(self.hparams.scaling_factors)
        assert cur_block < num_factors + 1

        # We're at the end of the Transformer, we don't scale at all at the end
        if cur_block == num_factors:
            return x

        scaler = self.scalers[cur_block]
        return scaler(x.movedim(-1, -2)).movedim(-2, -1)

    # Gives you both the block and (absolute) layer indices while iterating
    def enumerate_blocks_and_layers(self) -> Iterator:
        abs_layer_idx = 0
        layer_iter = iter(self.layers)
        for block_idx, block_size in enumerate(self.hparams.block_sizes):
            for _ in range(block_size):
                layer = next(layer_iter)
                yield block_idx, abs_layer_idx, layer

                abs_layer_idx += 1


class FunnelLayer(TransformerLayer):
    def __init__(self, hparams, q_stride: int, kv_stride: int):
        super().__init__(d_model=hparams.d_model, num_heads=hparams.num_heads, rel_pos_attn=True)

        self.q_stride = q_stride
        self.kv_stride = kv_stride
        self.hparams = hparams

    # Q is different from K and V right after pooling; K and V are always the same
    def forward(self, q: Tensor, kv: PaddedTensor, full_seq_len: int, context: PaddedTensor = None) -> PaddedTensor:
        pos_encodings = positional_encodings_for_strides(
            attn_type='rel_shift',
            q_stride=self.q_stride,
            k_stride=self.kv_stride,
            seq_len=full_seq_len,
            d_model=kv.shape[-1],
            device=kv.device,
        )
        return super().forward(q, kv, context, pos_enc=pos_encodings)


# Returns a Tensor or a list of Tensors containing positional encodings appropriate for the given strides.
@lru_cache(maxsize=10)
def positional_encodings_for_strides(attn_type: str, q_stride: int, k_stride: int, seq_len: int, d_model: int,
                                     device: torch.device) -> Union[Tensor, List[Tensor]]:
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
    elif attn_type == 'absolute':
        q_encodings = k_encodings = [base_encodings]
        q_encodings = [x[::q_stride] for x in q_encodings]
        k_encodings = [x[::k_stride] for x in k_encodings]

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

    elif attn_type == 'absolute':
        sines, cosines = get_sinusoidal_encodings(0, seq_len)
        return torch.cat([sines, cosines], dim=-1)
