from .AttentionState import AttentionState
from .ops import RelativePositionalAttention
from .ops import PositionwiseFFN
from .Performers import PerformerAttention
from .RemoteModels import *
from .Utilities import *
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from typing import *
import logging
import torch
import torch.nn as nn


@dataclass
class FunnelTransformerHparams:
    block_sizes: Sequence[int] = (4, 4, 4)
    d_model: int = 768
    num_heads: int = 12
    scaling_factors: Sequence[int] = (2, 2)

    # If None, d_embedding is set to equal d_model. For the generator in ELECTRA pretraining they are different.
    d_embedding: Optional[int] = None
    vocab_size: int = 30522
    attention_dropout: float = 0.1
    dropout: float = 0.1
    ffn_dropout: float = 0.0
    layer_norm_eps: float = 1e-9
    separate_cls: bool = True
    num_classes: int = 0

    positional_encoding_type: str = 'rel_shift'  # 'absolute', 'absolute_decoupled', 'rel_shift' or 'factorized'

    # Whether to return the pre-pooling output of each block
    return_block_outputs: bool = False
    use_performer_attention: bool = False
    use_initialization_scaling: bool = False
    upsampling: bool = False  # True for the "reverse" funnel transformer; e.g. a VAE decoder

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "num_heads must divide d_model evenly"
        if self.use_performer_attention:
            assert self.positional_encoding_type not in ('rel_shift', 'factorized'),\
                "Performer attention not supported with relative positional encodings"

        if not self.d_embedding:
            self.d_embedding = self.d_model

# Returned by FunnelTransformer.forward()
@dataclass
class FunnelTransformerOutput:
    final_state: Optional[Tensor] = None
    embedded_input: Optional[Tensor] = None
    hidden_states: List[Tensor] = field(default_factory=list)  # May be empty list
    logits: Optional[Tensor] = None


class FunnelTransformer(nn.Module):
    def __init__(self, hparams: Union[FunnelTransformerHparams, OmegaConf]):
        super().__init__()

        if isinstance(hparams, FunnelTransformerHparams):
            hparams = OmegaConf.structured(hparams)

        self.hparams = hparams

        if not hparams.upsampling:
            input_modules = [
                nn.Embedding(hparams.vocab_size, hparams.d_embedding),
                nn.LayerNorm(hparams.d_model, eps=hparams.layer_norm_eps),
                nn.Dropout(hparams.dropout)
            ]

            # If the embeddings have a different dimensionality from the Transformer hidden states,
            # we need to project the embeddings into d_model dimensions. This is needed for ELECTRA
            # pretraining where the generator has 4 times smaller hidden states than the discriminator,
            # but shares embeddings with the discriminator.
            if hparams.d_embedding != hparams.d_model:
                input_projection = nn.Linear(hparams.d_embedding, hparams.d_model)
                input_modules.insert(1, input_projection)

            self.input_layer = nn.Sequential(*input_modules)
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.vocab_size),
                nn.LogSoftmax(dim=-1)
            )

        self.layers = nn.ModuleList([
            FunnelLayer(hparams)
            for _ in range(sum(hparams.block_sizes))
        ])
        self.attention_state = None  # Lazily loaded

    # Vanilla function wrapper for forward_coroutine()
    def forward(self, x: Tensor, padding_mask: Tensor, **kwargs) -> FunnelTransformerOutput:
        coroutine = self.forward_coroutine(x, padding_mask, **kwargs)
        return next(coroutine)[-1]

    # Yields specified hidden states as they are generated while processing the input x, along with the absolute indices
    # of the Transformer layers that produced them. The consumer of this coroutine is allowed to send back transformed
    # versions of these hidden states, which are then used as the input to the next layer in the Transformer.
    def forward_coroutine(self, x: Tensor, padding_mask: Tensor, *,
                          return_embedded_input: bool = False,
                          return_logits: bool = True,
                          states_to_yield: Optional[List[int]] = None):
        hparams = self.hparams

        if not hparams.upsampling:
            x = self.input_layer(x)  # x.shape == (batch, length, d_model)

        # This lazy loading allows the user to give us a pre-initialized, possibly shared AttentionState object
        if not (attn_state := self.attention_state):
            attn_state = self.attention_state = AttentionState(hparams)

        # If the AttentionState object is shared, then it's not FunnelTransformer's responsibility to configure it
        # for the current input, because that could result in it getting configured twice
        if not attn_state.shared:
            attn_state.configure_for_input(x.shape[1], x.dtype, x.device, padding_mask)

        hidden_states = []
        layer_iter = iter(enumerate(self.layers))
        q = kv = x

        for block_idx, block_size in enumerate(self.hparams.block_sizes):
            for rel_layer_idx in range(block_size):
                # Let AttentionState know we're starting a new block
                attn_state.block_begin_flag = (rel_layer_idx == 0)
                abs_layer_idx, layer = next(layer_iter)
                cross_attn_keys = None

                if states_to_yield and abs_layer_idx in states_to_yield:
                    maybe_transformed_x = yield block_idx, abs_layer_idx, q

                    # The consumer of this generator may or may not send us anything
                    if maybe_transformed_x is not None:
                        # The consumer may have sent us a tuple of the form (qkv, cross attention target)
                        if isinstance(maybe_transformed_x, tuple):
                            cross_attn_keys = maybe_transformed_x[1]
                            maybe_transformed_x = maybe_transformed_x[0]

                        q = kv = maybe_transformed_x
                        yield  # Don't return anything from the .send() method

                q = kv = layer(q, kv, attn_state, cross_attn_keys=cross_attn_keys)
                attn_state.block_begin_flag = False

            q = attn_state.maybe_scale_input(kv)

            # When we're upsampling, there's no stage where we scale Q but don't scale K and V.
            if self.hparams.upsampling:
                kv = q

            # Cache block outputs if indicated
            if hparams.return_block_outputs:
                hidden_states.append(kv)

        output = FunnelTransformerOutput(
            final_state=q,
            hidden_states=hidden_states
        )
        if return_embedded_input:
            output.embedded_input = x
        if return_logits:
            assert hasattr(self, 'output_layer'), "Can't return logits if we don't have an output Linear layer"
            # Non-autoregressively generate a softmax distribution over words
            output.logits = self.output_layer(q)

        # Last thing we yield is the final output
        yield -1, -1, output

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
        self.input_layer.load_state_dict({
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
            attn_type='positional_encoding_type',
            num_class='num_classes',
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
    def __init__(self, hparams):
        super().__init__()

        d_model = hparams.d_model
        if hparams.positional_encoding_type == 'absolute':
            # Softmax attention with absolute, sinusoidal positional encodings
            if not hparams.use_performer_attention:
                raw_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=hparams.num_heads)

            # Performer attention with absolute positional embeddings
            else:
                raw_attn = PerformerAttention(**select(hparams, 'd_model', 'num_heads'))

            # Wrap the raw attention module with this function that adds the positional encodings to the queries and
            # keys, but not to the values, as proposed in the Shortformer paper
            def absolute_pos_attn_func(q: Tensor, k: Tensor, v: Tensor, attn_state: AttentionState):
                q_pos_encodings, k_pos_encodings = attn_state.get_positional_encodings()
                q = q + q_pos_encodings
                k = k + k_pos_encodings
                return raw_attn(q, k, v)

            self.attention = absolute_pos_attn_func

        # Either softmax or Performer attention with relative positional embeddings
        else:
            self.attention = RelativePositionalAttention(hparams)

        self.cross_attention = None
        self.hparams = hparams
        self.feedforward = PositionwiseFFN(hparams.d_model, hparams.d_model * 4, hparams.dropout, hparams.ffn_dropout,
                                           layer_norm_eps=hparams.layer_norm_eps)

    @property
    def use_cross_attention(self) -> bool:
        return bool(self.cross_attention)

    @use_cross_attention.setter
    def use_cross_attention(self, value: bool):
        self.cross_attention = RelativePositionalAttention(self.hparams) if value else None

    # Q is different from K and V right after pooling; K and V are always the same
    def forward(self, q: Tensor, kv: Tensor, attn_state: AttentionState, cross_attn_keys: Tensor = None) -> Tensor:
        # These custom attention and feedforward layers have built-in residual connections
        q = self.attention(q, kv, kv, attn_state)
        if cross_attn_keys is not None:
            q = self.cross_attention(q, cross_attn_keys, cross_attn_keys, attn_state)

        q = self.feedforward(q)
        return q
