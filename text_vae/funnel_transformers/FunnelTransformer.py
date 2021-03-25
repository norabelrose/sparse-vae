from .AttentionState import *
from .FunnelOps import AttentionHparams, PositionwiseFFN, FunnelAttention
from .RemoteModels import *
from ..core import PaddedTensor, PositionalEncoding
from ..core.Utilities import *
from dataclasses import dataclass, field
from numpy import prod
from omegaconf import OmegaConf
from pytorch_lightning.utilities import AttributeDict
from torch import nn, Tensor
from typing import *
import logging
import torch


@dataclass
class FunnelTransformerHparams(AttentionHparams):
    block_sizes: Sequence[int] = (4, 4, 4)
    scaling_factors: Sequence[int] = (2, 2)

    # If None, d_embedding is set to equal d_model. For the generator in ELECTRA pretraining they are different.
    d_embedding: Optional[int] = None
    max_seq_length: Optional[int] = None
    padding_idx: Optional[int] = 0
    vocab_size: int = 30522

    # Whether to return the pre-pooling output of each block
    return_block_outputs: bool = True
    use_embedding: bool = True

    # Use strided convolutions instead of average pooling for downsampling, and
    # transpose convolutions instead of nearest-neighbor upsampling
    use_convolutions: bool = True
    upsampling: bool = False  # True for the "reverse" funnel transformer; e.g. a VAE decoder

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "num_heads must divide d_model evenly"


# Returned by FunnelTransformer.forward()
@dataclass
class FunnelTransformerOutput:
    original_ids: Tensor
    final_state: Tensor
    hidden_states: List[PaddedTensor] = field(default_factory=list)  # May be empty list

    embedded_input: Optional[Tensor] = None


class FunnelTransformer(nn.Module):
    def __init__(self, hparams: Union[FunnelTransformerHparams, DictConfig]):
        super().__init__()

        if isinstance(hparams, FunnelTransformerHparams):
            hparams = OmegaConf.structured(hparams)

        if not hparams.d_embedding:
            hparams.d_embedding = hparams.d_model

        self.hparams = hparams

        max_len = hparams.max_seq_length
        if max_len:
            if hparams.upsampling:
                max_len //= prod(hparams.scaling_factors)

            self.abs_pos_encodings = PositionalEncoding(max_len, hparams.d_model)
        else:
            self.abs_pos_encodings = None

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
            input_modules = [
                nn.Embedding(hparams.vocab_size, hparams.d_embedding, padding_idx=hparams.padding_idx),
                nn.LayerNorm(hparams.d_embedding, eps=hparams.layer_norm_eps),
                nn.Dropout(hparams.dropout)
            ]

            # If the embeddings have a different dimensionality from the Transformer hidden states,
            # we need to project the embeddings into d_model dimensions. This is needed for ELECTRA
            # pretraining where the generator has 4 times smaller hidden states than the discriminator,
            # but shares embeddings with the discriminator.
            if hparams.d_embedding != hparams.d_model:
                input_projection = nn.Linear(hparams.d_embedding, hparams.d_model)
                input_modules.insert(2, input_projection)

            self.input_layer = nn.Sequential(*input_modules)

        self.layers = nn.ModuleList([
            FunnelLayer(hparams)
            for _ in range(sum(hparams.block_sizes))
        ])
        self._attention_state = None

    @property
    def attention_state(self) -> AttentionState:
        if not self._attention_state:
            self._attention_state = AttentionState(self.hparams)

        return self._attention_state

    @attention_state.setter
    def attention_state(self, value: AttentionState):
        self._attention_state = value

    def layer_with_indices(self, block_idx: int, layer_idx: int) -> 'FunnelLayer':
        layer_iter = iter(self.layers)
        for i, block_size in enumerate(self.hparams.block_sizes):
            for j in range(block_size):
                layer = next(layer_iter)
                if i == block_idx and j == layer_idx:
                    return layer

        raise ValueError

    # Vanilla function wrapper for forward_coroutine()
    def forward(self, x: Tensor, **kwargs) -> FunnelTransformerOutput:
        coroutine = self.forward_coroutine(x, **kwargs)
        return [x for x in coroutine][-1][-1]  # noqa

    # Yields specified hidden states as they are generated while processing the input x, along with the absolute indices
    # of the Transformer layers that produced them. The consumer of this coroutine is allowed to send back transformed
    # versions of these hidden states, which are then used as the input to the next layer in the Transformer.
    def forward_coroutine(self, x: Tensor, padding_mask: Optional[Tensor] = None, segment_ids: Optional[Tensor] = None,
                          start_block: int = 0, end_block: int = None):
        attn_state = self.attention_state
        hparams = self.hparams
        original = x

        if not hparams.upsampling and hparams.use_embedding:
            x = self.input_layer(x)  # x.shape == (...length, d_model)

        if self.abs_pos_encodings:
            x = self.abs_pos_encodings(x)

        # If the AttentionState object is shared, then it's not FunnelTransformer's responsibility to configure it
        # for the current input, because that could result in it getting configured twice
        if not attn_state.shared:
            attn_state.configure_for_input(x.shape[-2], padding_mask, segment_ids)

        hidden_states = []
        context = None

        blocks = self.hparams.block_sizes
        start_layer = sum(blocks[:start_block])
        layer_iter = iter(self.layers[start_layer:])

        num_blocks = len(blocks)
        end_block = end_block % num_blocks if end_block else num_blocks
        last_block = num_blocks - 1

        attn_state.current_block = start_block
        q = kv = x

        for block_idx, block_size in enumerate(self.hparams.block_sizes[start_block:end_block], start=start_block):
            padding = attn_state.padding_mask_with_length(kv.shape[-2])
            mask = padding[:, None, None, :] if padding is not None else None

            for rel_layer_idx in range(block_size):
                # Let AttentionState know we're starting a new block
                attn_state.block_begin_flag = (rel_layer_idx == 0)
                layer = next(layer_iter)

                q = kv = layer(q, kv, mask, attn_state, context=context)
                if mask.shape[-1] != kv.shape[-2]:
                    padding = attn_state.padding_mask_with_length(kv.shape[-2])
                    mask = padding[:, None, None, :] if padding is not None else None

                attn_state.block_begin_flag = False

            # Cache block outputs if indicated
            if hparams.return_block_outputs:
                hidden_states.append(PaddedTensor(data=kv, padding=padding))

            # We don't need to do any of this stuff if we just finished the last block
            if block_idx == last_block:
                break

            q = self.maybe_scale_hidden(kv)

            # The consumer of this generator may or may not send us anything
            maybe_transformed_q = yield block_idx, q
            if maybe_transformed_q is not None:
                # The consumer of this generator can send us tensors to cross-attend to
                if isinstance(maybe_transformed_q, tuple):
                    context = maybe_transformed_q[1]
                    maybe_transformed_q = maybe_transformed_q[0]
                else:
                    context = None

                q = kv = maybe_transformed_q
                yield  # Don't return anything from the .send() method

            # When we're upsampling, there's no stage where we scale Q but don't scale K and V.
            if self.hparams.upsampling:
                kv = q

        # Last thing we yield is the final output
        yield -1, FunnelTransformerOutput(
            original_ids=original,
            final_state=q,
            embedded_input=x,
            hidden_states=hidden_states
        )

    # Either downsample or upsample the tensor, whichever is appropriate for the current block.
    def maybe_scale_hidden(self, x: Tensor) -> Tensor:
        attn_state = self.attention_state

        # Sanity check
        cur_block = attn_state.current_block
        num_factors = len(attn_state.scaling_factors)
        assert cur_block < num_factors + 1

        # We're at the end of the Transformer, we don't scale at all at the end
        if cur_block == num_factors:
            return x

        attn_state.current_block = cur_block + 1

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
    def __init__(self, hparams, causal: bool = False, use_cross_attention: bool = False):
        super().__init__()

        # d_model = hparams.d_model
        self.attention = FunnelAttention(hparams)
        self.causal = causal
        self.cross_attention = use_cross_attention
        self.hparams = hparams
        self.feedforward = PositionwiseFFN(hparams.d_model, hparams.d_model * 4, hparams.dropout, hparams.ffn_dropout,
                                           layer_norm_eps=hparams.layer_norm_eps)

    @property
    def use_cross_attention(self):
        return bool(self.cross_attention)

    @use_cross_attention.setter
    def use_cross_attention(self, value: bool):
        self.cross_attention = FunnelAttention(self.hparams) if value else None

    # Q is different from K and V right after pooling; K and V are always the same
    def forward(self, q: Tensor, kv: Tensor, mask: Tensor, attn_state: AttentionState, context: Tensor = None) -> Tensor:
        pos_encodings = attn_state.get_positional_encodings(kv.shape[-1], q.device)

        # These custom attention and feedforward layers have built-in residual connections
        q = self.attention(q, kv, kv, mask=mask, pos_encodings=pos_encodings)
        if context is not None:
            ctx_len = context.shape[-2]
            mask = attn_state.padding_mask_with_length(ctx_len)[:, None, None, :] if context is not None else None
            q = self.cross_attention(q, context, context, mask=mask, pos_encodings=pos_encodings)  # noqa

        q = self.feedforward(q)
        return q
