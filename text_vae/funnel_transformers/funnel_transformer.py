from .funnel_ops import AttentionHparams, PositionwiseFFN, FunnelAttention
from .remote_models import *
from ..core import PaddedTensor
from ..core.utilities import *
from dataclasses import dataclass
from functools import lru_cache
from numpy import cumprod, prod
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities import AttributeDict
from torch import nn, Tensor
from typing import *
import logging
import torch


@dataclass
class FunnelTransformerHparams(AttentionHparams):
    block_sizes: Sequence[int] = (4, 4, 4)
    scaling_factors: Sequence[int] = (2, 4)

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
                nn.LayerNorm(hparams.d_model, eps=hparams.layer_norm_eps),
                nn.Dropout(hparams.dropout)
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

    # Convenient method for loading old checkpoints
    def enumerate_parameters_by_layer(self) -> Iterator[Tuple[str, torch.nn.Parameter, int]]:
        for block_idx, abs_layer_idx, layer in self.enumerate_blocks_and_layers():
            for var_name, param in layer.named_parameters():
                yield var_name, param, abs_layer_idx

    def path_to_pretrained_checkpoint(self) -> Path:
        url = remote_model_url_for_hparams(self.hparams, suffix="-PT")
        return load_remote_model(url)

    def load_pretrained_weights(self, freeze: bool = True, verbose: bool = False):
        model_path = self.path_to_pretrained_checkpoint()

        # Our parameter names will look like this: 'blocks.0.layers.2.attention.v_head.bias', but the training
        # files will have the form 'attn_layers.2.v_head.bias'. We need to convert here.
        state_dict = torch.load(str(model_path / "model.pt"))
        noninitialized_keys = []

        # Don't forget about the embeddings
        assert len(self.input_layer) == 3   # We don't support loading PyTorch weights where d_embedding != d_model
        self.input_layer.load_state_dict({  # noqa
            '0.weight': state_dict['input_layer.0.lookup_table'],
            '1.weight': state_dict['input_layer.1.weight'],
            '1.bias': state_dict['input_layer.1.bias']
        }, strict=True)
        self.input_layer.requires_grad_(not freeze)

        for var_name, param, absolute_index in self.enumerate_parameters_by_layer():
            keys = var_name.split('.')
            keys[0] = replace_all(keys[0], {  # attention.v_head.bias -> attn_layers.v_head.bias
                'attention': 'attn_layers',
                'feedforward': 'pffn_layers'
            })

            keys.insert(1, str(absolute_index))  # attn_layers.v_head.bias -> attn_layers.2.v_head.bias
            old_name = '.'.join(keys)

            try:
                old_weights: Tensor = state_dict[old_name]

                if old_weights.shape != param.data.shape:
                    if "r_kernel" in var_name:
                        old_weights = old_weights.permute(1, 0, 2)
                    else:
                        old_weights = old_weights.reshape(*param.data.shape)

                param.data = old_weights
                param.requires_grad = not freeze
            except KeyError:
                noninitialized_keys.append({'new_name': var_name, 'old_name': old_name})

        if len(noninitialized_keys) > 0 and verbose:
            logger = logging.getLogger(__name__)
            logger.warning(f'Failed to initialize weights: {noninitialized_keys}')

    # For the "args" parameter in the old FunnelTFM.__init__()
    def get_backward_compatible_args(self) -> AttributeDict:
        return transmute(
            self.hparams,
            attn_type='positional_attention_type',
            num_class='0',
            pad_id='None',
            seg_id_cls='2',
            truncate_seq='True'
        )
    
    # Get a dictionary compatible with the old ModelConfig class from Funnel-Transformers
    def get_backward_compatible_dict(self) -> Dict:
        return transmute(
            self.hparams,
            'vocab_size', 'd_model', 'dropout', 'separate_cls',
            d_embed='d_model',
            n_head='num_heads',
            d_head='d_model // num_heads',
            d_inner='d_model * 4',
            dropatt='attention_dropout',
            dropact='ffn_dropout',
            block_size="'_'.join([str(x) for x in block_sizes])",
            
            # We lose info here since Funnel-Transformers doesn't support different scaling factors for each block
            pooling_size='scaling_factors[0]',
            pooling_type='"mean"',
            pool_q_only='True'
        )


class FunnelLayer(nn.Module):
    def __init__(self, hparams, q_stride: int, kv_stride: int):
        super().__init__()

        # d_model = hparams.d_model
        self.attention = FunnelAttention(hparams)
        self.q_stride = q_stride
        self.kv_stride = kv_stride
        self.hparams = hparams
        self.cross_attention = None
        self.feedforward = PositionwiseFFN(hparams.d_model, hparams.d_model * 4, hparams.dropout, hparams.ffn_dropout,
                                           layer_norm_eps=hparams.layer_norm_eps)

    @property
    def use_cross_attention(self):
        return bool(self.cross_attention)

    @use_cross_attention.setter
    def use_cross_attention(self, value: bool):
        self.cross_attention = FunnelAttention(self.hparams) if value else None

    # Q is different from K and V right after pooling; K and V are always the same
    def forward(self, q: Tensor, kv: PaddedTensor, full_seq_len: int, context: PaddedTensor = None) -> PaddedTensor:
        pos_encodings = positional_encodings_for_strides(
            attn_type=self.hparams.positional_attention_type,
            q_stride=self.q_stride,
            k_stride=self.kv_stride,
            seq_len=full_seq_len,
            d_model=kv.shape[-1],
            device=kv.device,
            separate_cls=self.hparams.separate_cls
        )

        # These custom attention and feedforward layers have built-in residual connections
        q = self.attention(q, kv, kv, pos_encodings=pos_encodings)
        if context is not None and self.cross_attention is not None:
            q = self.cross_attention(q, context, context, pos_encodings=pos_encodings)  # noqa

        q = self.feedforward(q)
        return q


# Returns a Tensor or a list of Tensors containing positional encodings appropriate for the given strides.
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
