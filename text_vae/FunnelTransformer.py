from .AttentionState import AttentionState
from .ops import RelativePositionalAttention
from .ops import LayerNorm
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
    block_sizes: Tuple[int, ...] = (4, 4, 4)
    d_model: int = 768
    num_heads: int = 12
    scaling_factors: Tuple[int, ...] = (2, 2)

    # If None, d_embedding is set to equal d_model. For the generator in ELECTRA pretraining they are different.
    d_embedding: Optional[int] = None
    vocab_size: int = 30522
    attention_dropout: float = 0.1
    dropout: float = 0.1
    ffn_dropout: float = 0.0
    separate_cls: bool = True
    num_classes: int = 0

    positional_encoding_type: str = 'rel_shift'  # 'absolute', 'absolute_decoupled', 'rel_shift' or 'factorized'
    rezero_nonpretrained_blocks: bool = False

    # Whether to return the pre-pooling output of each block on forward(). If a Sequence, then only the output of
    # selected blocks will be returned.
    block_outputs_to_return: Sequence[int] = field(default_factory=list)
    use_performer_attention: bool = False
    upsampling: bool = False  # True for the "reverse" funnel transformer; e.g. a VAE decoder

    def __post_init__(self):
        if self.use_performer_attention:
            assert self.positional_encoding_type not in ('rel_shift', 'factorized'),\
                "Performer attention not supported with relative positional encodings"

        # Make it so scaling_factors and block_sizes are equal length; last scaling factor is 1 (no scaling)
        if len(self.scaling_factors) < len(self.block_sizes):
            self.scaling_factors += (1,)

        if not self.d_embedding:
            self.d_embedding = self.d_model


class FunnelTransformer(nn.Module):
    def __init__(self, hparams: Union[FunnelTransformerHparams, OmegaConf],
                 shared_attention_state: Optional[AttentionState] = None):
        super().__init__()

        if isinstance(hparams, FunnelTransformerHparams):
            hparams = OmegaConf.structured(hparams)

        self.hparams = hparams

        if not hparams.upsampling:
            input_modules = [
                nn.Embedding(hparams.vocab_size, hparams.d_embedding),
                LayerNorm(hparams.d_model),
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

        self.blocks = nn.ModuleList([FunnelBlock(hparams, size) for size in hparams.block_sizes])
        self.attention_state = shared_attention_state or AttentionState(hparams)

    def forward(self, x: Dict[str, Any], reset_attention_state: bool = True) -> Dict[str, Any]:
        hparams = self.hparams

        if not hparams.upsampling:
            x['input'] = self.input_layer(x['input'])  # x.shape == (batch, length, d_model)
        
        attn_state = self.attention_state
        attn_state.configure_for_input(x['input'], x.get('input_mask'))

        x.update(q=x['input'], kv=x.pop('input'), attn_state=attn_state, hidden_states=[])
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Cache intermediate hidden states if indicated
            if i in hparams.block_outputs_to_return:
                x['hidden_states'].append(x['kv'])

        if hparams.upsampling:
            # Non-autoregressively generate a softmax distribution over words
            x['logits'] = self.output_layer(x.pop('q'))
        else:
            x['output'] = x.pop('q')

        del x['attn_state'], x['kv']

        if reset_attention_state:
            attn_state.reset()
        return x
    
    def enumerate_layers(self) -> Iterator:
        absolute_index = 0
        for block in self.blocks:
            for layer in block.layers:
                yield absolute_index, layer
                absolute_index += 1

    # Convenient method for loading old checkpoints
    def enumerate_parameters_by_layer(self) -> Iterator[Tuple[str, torch.nn.Parameter, int]]:
        for index, layer in self.enumerate_layers():
            for var_name, param in layer.named_parameters():
                yield var_name, param, index

    def path_to_pretrained_checkpoint(self) -> Path:
        url = remote_model_url_for_hparams(self.hparams, suffix="-PT")
        return load_remote_model(url)

    def load_pretrained_weights(self, verbose: bool = False):
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
                q += q_pos_encodings
                k += k_pos_encodings
                return raw_attn(q, k, v)

            self.attention = absolute_pos_attn_func

        # Either softmax or Performer attention with relative positional embeddings
        else:
            self.attention = RelativePositionalAttention(hparams)

        self.feedforward = PositionwiseFFN(hparams.d_model, hparams.d_model * 4, hparams.dropout, hparams.ffn_dropout)

        # A Callable (e.g. a Module) that will be called with the output of this layer. The output of
        # *this transform* will then be passed on to the next layer. This makes it possible to add extra information
        # (possibly stochastically) to the upsampling process.
        self.output_transform: Optional[Callable] = None

    # Q is different from K and V right after pooling; K and V are always the same
    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        # These custom attention and feedforward layers have built-in residual connections
        x['kv'] = self.attention(x['q'], x['kv'], x['q'], x['attn_state'])
        x['kv'] = self.feedforward(x['kv'])

        if self.output_transform:
            # The transform may add or mutate its own keys in the dictionary (see i.e. Autoencoder)
            x = self.output_transform(x)

        return x


class FunnelBlock(nn.Module):
    def __init__(self, hparams, num_layers: int):
        super().__init__()

        self.layers = nn.ModuleList([FunnelLayer(hparams) for _ in range(num_layers)])
        self.rezero_alpha = None

    def activate_rezero(self):
        self.rezero_alpha = nn.Parameter(torch.tensor(0))

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        for i, layer in enumerate(self.layers):
            x['attn_state'].begin_block_flag = (i == 0)  # Let AttentionState know we're starting a new block
            x = layer(x)
            x['attn_state'].begin_block_flag = False

        # With ReZero, we introduce an additional residual connection between blocks, where the output of each
        # block is multiplied by a parameter alpha that is initialized to zero. When alpha == 0, the block has
        # 'no effect' and simply outputs an average pooled version of its input.
        if self.rezero_alpha is not None:
            x['kv'] = x['q'] + (x['kv'] * self.rezero_alpha)

        x['q'] = x['attn_state'].scale_input(x['kv'])
        return x
