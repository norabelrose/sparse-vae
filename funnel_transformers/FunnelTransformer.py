from __future__ import annotations

from .AttentionState import AttentionState
from .RelativePositionalAttention import LayerNorm
from .RelativePositionalAttention import RelativePositionalAttention
from .RelativePositionalAttention import PositionwiseFFN
from .RemoteModels import *
from ..HparamUtils import *
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from typing import *
import logging
import torch
import torch.nn as nn


class FunnelTransformer(nn.Module):
    default_hparams = AttributeDict(
        block_sizes=(4, 4, 4),
        d_model=768,
        num_heads=12,
        scaling_factors=(2, 2),

        # If None, d_embedding is set to equal d_model. For the generator in ELECTRA pretraining they are different.
        d_embedding=None,
        vocab_size=30522,
        attention_dropout=0.1,
        dropout=0.1,
        ffn_dropout=0.0,
        pooling_type='mean',
        separate_cls=True,
        pool_q_only=True,
        truncate_seq=True,
        seg_id_cls=2,  # Segment ID of the [CLS] token
        pad_id=None,
        num_classes=0,

        attention_type='rel_shift',
        rezero_nonpretrained_blocks=False,
    
        # Whether to return the pre-pooling output of each block on forward(). If a Sequence, then only the output of
        # selected blocks will be returned.
        return_block_outputs=False,
        use_performer_attention=False,
        upsampling=False,  # True for the "reverse" funnel transformer; e.g. a VAE decoder

        has_decoder_block=False,        # Set to True by FunnelForPreTraining and used by AttentionState.
        use_mlm_head=False
    )
    
    def __init__(self, hparams: MutableMapping[str, Any], shared_attention_state: Optional[AttentionState] = None):
        super().__init__()

        if hparams.get('use_performer_attention'):
            assert 'attention_type' not in hparams or hparams['attention_type'] == 'factorized',\
                "Performer attention is not compatible with the relative shift method of relative positional attention."
            hparams['attention_type'] = 'factorized'

        hparams = merge(self.default_hparams, hparams)

        # Make it so scaling_factors and block_sizes are equal length; last scaling factor is 1 (no scaling)
        if len(hparams.scaling_factors) < len(hparams.block_sizes):
            hparams.scaling_factors += (1,)

        if not hparams.d_embedding:
            hparams.d_embedding = hparams.d_model

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

    def forward(self, x: Tensor, input_mask: Tensor = None, seg_id: Tensor = None) -> Dict[str, Any]:
        config = self.hparams
        hidden_states = []

        if not config.upsampling:
            x = self.input_layer(x)  # x.shape == (batch, length, d_model)
        
        attn_state = self.attention_state
        attn_state.configure_for_input(x, input_mask, seg_id)

        q = kv = x
        for i, block in enumerate(self.blocks):
            q, kv = block(q, kv, attn_state)

            # Cache intermediate hidden states if indicated
            return_blocks = config.return_block_outputs
            return_blocks = return_blocks if type(return_blocks) == bool else i in return_blocks
            if return_blocks:
                hidden_states.append(kv)

        output = {}
        if config.upsampling:
            # Non-autoregressively generate a softmax distribution over words
            output['logits'] = self.output_layer(q)
        else:
            output['output'] = q

        if len(hidden_states) > 0:
            output['hidden_states'] = hidden_states

        attn_state.reset()
        return output
    
    def enumerate_layers(self) -> Iterator[int, FunnelLayer]:
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

    def load_pretrained_weights(self):
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

                # The old Funnel-Transformer had custom Dense layers that reshaped the input and therefore had
                # 3D kernel tensors- in this project we're more conventional so we're using standard nn.Linear modules
                if old_weights.shape != param.data.shape:
                    if "r_kernel" in var_name:
                        old_weights = old_weights.permute(1, 0, 2)
                    else:
                        old_weights = old_weights.reshape(*param.data.shape)

                param.data = old_weights
            except KeyError:
                noninitialized_keys.append({'new_name': var_name, 'old_name': old_name})

        if len(noninitialized_keys) > 0:
            logger = logging.getLogger(__name__)
            logger.warning(f'PretrainedModelManager: Failed to initialize weights: {noninitialized_keys}')

    # For the "args" parameter in the old FunnelTFM.__init__()
    def get_backward_compatible_args(self) -> AttributeDict:
        return transmute(self.hparams, 'pad_id', 'seg_id_cls', 'truncate_seq',
                         attn_type='attention_type', num_class='num_classes')
    
    # Get a dictionary compatible with the old ModelConfig class from Funnel-Transformers
    def get_backward_compatible_dict(self) -> Dict:
        return transmute(
            self.hparams,
            'vocab_size', 'd_model', 'dropout', 'pooling_type', 'separate_cls', 'pool_q_only',
            d_embed='d_model',
            n_head='num_heads',
            d_head='d_model // num_heads',
            d_inner='d_model * 4',
            dropatt='attention_dropout',
            dropact='ffn_dropout',
            block_size="'_'.join([str(x) for x in block_sizes])",
            
            # We lose info here since Funnel-Transformers doesn't support different scaling factors for each block
            pooling_size='scaling_factors[0]'
        )


class FunnelLayer(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.attention = RelativePositionalAttention(hparams)
        self.feedforward = PositionwiseFFN(hparams.d_model, hparams.d_model * 4, hparams.dropout, hparams.ffn_dropout)

        # A Callable (e.g. a Module) that will be called with the output of this layer. The output of
        # *this transform* will then be passed on to the next layer. This makes it possible to add extra information
        # (possibly stochastically) to the upsampling process.
        self.output_transform: Optional[Callable] = None

    # Q is different from K and V right after pooling; K and V are always the same
    def forward(self, q, kv, attention_state: AttentionState):
        # These custom attention and feedforward layers have built-in residual connections
        x = self.attention(q, kv, kv, attention_state)
        x = self.feedforward(x)

        if self.output_transform:
            x = self.output_transform(x)

        return x


class FunnelBlock(nn.Module):
    def __init__(self, hparams, num_layers: int):
        super().__init__()

        self.layers = nn.ModuleList([FunnelLayer(hparams) for _ in range(num_layers)])
        self.rezero_alpha = None

    def activate_rezero(self):
        self.rezero_alpha = nn.Parameter(torch.tensor(0))

    def forward(self, q, kv, attention_state: AttentionState) -> Tuple[Tensor, Tensor]:
        with attention_state.begin_block():
            kv = self.layers[0](q, kv, attention_state)

        for layer in self.layers[1:]:
            kv = layer(kv, kv, attention_state)

        # With ReZero, we introduce an additional residual connection between blocks, where the output of each
        # block is multiplied by a parameter alpha that is initialized to zero. When alpha == 0, the block has
        # 'no effect' and simply outputs an average pooled version of its input.
        if self.rezero_alpha is not None:
            kv = q + (kv * self.rezero_alpha)

        q = attention_state.scale_input(kv)
        return q, kv
