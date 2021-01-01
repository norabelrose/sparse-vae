from __future__ import annotations

from .AttentionState import AttentionState
from copy import deepcopy
from .RelativePositionalAttention import LayerNorm
from .RelativePositionalAttention import RelativePositionalAttention
from .RelativePositionalAttention import PositionwiseFFN
from .RemoteModels import *
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from typing import *
import logging
import torch
import torch.nn as nn


class FunnelTransformer(nn.Module):
    default_hparams = dict(
        block_sizes=(4, 4, 4),
        d_model=768,
        num_heads=12,

        # **Default values taken from pretrained config files & flags**
        vocab_size=30522,
        attention_dropout=0.1,
        dropout=0.1,
        ffn_dropout=0.0,
        pooling_type='mean',
        scaling_factors=2,
        separate_cls=True,
        pool_q_only=True,
        truncate_seq=True,
        seg_id_cls=2,  # Segment ID of the [CLS] token
        pad_id=None,
        num_classes=0,
        use_classification_head=False,

        attention_type='rel_shift',
        rezero_blocks=(),  # Blocks for which to use ReZero
        max_position_embeddings=512,
        return_attention_state=False,  # Useful for a VAE encoder; can reuse the state in the decoder
    
        # Whether to return the pre-pooling output of each block on forward(). If a Sequence, then only the output of
        # selected blocks will be returned.
        return_block_outputs=False,
        use_performer_attention=False,
        upsampling=False,  # True for the "reverse" funnel transformer; e.g. a VAE decoder

        has_decoder_block=False,        # Set to True by FunnelForPreTraining and used by AttentionState.
        use_mlm_head=False
    )
    
    def __init__(self, **kwargs):
        super().__init__()
        
        hparams = AttributeDict(**self.default_hparams, **kwargs)

        # Turn a single floating point scaling factor x into (x, x, x...) of the appropriate length
        if isinstance(hparams.scaling_factors, int):
            factor = hparams.scaling_factors
            hparams.scaling_factors = tuple(factor for _ in range(len(hparams.block_sizes) - 1))

        # Make it so scaling_factors and block_sizes are equal length; last scaling factor is 1 (no scaling)
        if len(hparams.scaling_factors) < len(hparams.block_sizes):
            hparams.scaling_factors += (1,)

        self.hparams = hparams

        if not hparams.upsampling:
            self.input_layer = nn.Sequential(
                nn.Embedding(hparams.vocab_size, hparams.d_model),
                LayerNorm(hparams.d_model),
                nn.Dropout(hparams.dropout))
        else:
            self.output_linear = nn.Linear(hparams.d_model, hparams.vocab_size)

        self.blocks = nn.ModuleList([FunnelBlock(hparams, size) for size in hparams.block_sizes])
        for block_index in hparams.rezero_blocks:
            self.blocks[block_index].activate_rezero()

        self.attention_state = AttentionState(hparams)

        if hparams.use_classification_head:
            self.cls_head = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_model),
                nn.Tanh(),
                nn.Dropout(hparams.dropout),
                nn.Linear(hparams.d_model, self.hparams.num_classes))
            self.cls_loss = nn.CrossEntropyLoss()

    # Returns a copy of the transformer whose upsampling parameter is flipped
    def inverted_copy(self, reverse_layer_order: bool = True, reinitialize_blocks=()):
        new_funnel: FunnelTransformer = deepcopy(self)
        new_funnel.config.upsampling = not self.hparams.upsampling

        for block in reinitialize_blocks:
            for param in block.parameters():
                # Glorot uniform initialization
                dim = min(param.data.dim() - 1, 1)
                stdv = 1. / param.data.size(dim) ** 0.5

                param.data.uniform_(-stdv, stdv)

        if reverse_layer_order:
            for block in new_funnel.blocks:
                block.layers.reverse()

            new_funnel.blocks.reverse()

        return new_funnel

    # All inputs should be of shape (batch, length)
    def forward(self, x: Tensor, input_mask: Tensor = None, seg_id: Tensor = None, cls_target: Tensor = None):
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
            if (type(return_blocks) == bool and return_blocks) or i in return_blocks:
                hidden_states.append(kv)

        # Non-autoregressively generate a softmax distribution over words
        if config.upsampling:
            q = self.output_linear(q)
            q = nn.functional.softmax(q, dim=-1)

        # We're returning all hidden states as a list
        output = []
        if len(hidden_states) == len(self.blocks):
            output.append(hidden_states)

        # We're returning the last hidden state, and a subset of the intermediate states
        elif len(hidden_states) > 0:
            output.extend([q, hidden_states])

        # We're only returning the last hidden state
        else:
            output.append(q)

        if cls_target is not None:
            ret_dict = {}

            last_hidden = q[-1][:, 0]
            cls_logits = self.cls_head(last_hidden)
            prediction = torch.argmax(cls_logits, -1)
            ret_dict["cls_pred"] = prediction
            cls_loss = self.cls_loss(cls_logits, cls_target)
            ret_dict["cls_loss"] = cls_loss
            cls_correct = prediction == cls_target
            cls_correct = cls_correct.type(torch.float32).sum()
            ret_dict["cls_corr"] = cls_correct

            output.append(ret_dict)

        if config.return_attention_state:
            attn_state.reset(keep_masks=True)
            output.append(attn_state)
        else:
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
        block_size_to_name = {
            (4, 4, 4): "B4-4-4H768-ELEC",
            (6, 6, 6): "B6-6-6H768-ELEC",
            (8, 8, 8): "B8-8-8H1024-ELEC",
            (10, 10, 10): "B10-10-10H1024-ELEC"
        }
        block_size_to_dims = {
            (4, 4, 4): (768, 12),  # d_model and num_heads
            (6, 6, 6): (768, 12),
            (8, 8, 8): (1024, 16),
            (10, 10, 10): (1024, 16)
        }

        # Sanity checks
        beginning_blocks = self.hparams.block_sizes[0:3]
        pretrained_d_model = block_size_to_dims[beginning_blocks][0]
        assert len(self.hparams.block_sizes) >= 3
        assert beginning_blocks in block_size_to_name, f"No pretrained model with block layout {beginning_blocks}"
        assert self.hparams.d_model == pretrained_d_model, \
            f"Pretrained model {block_size_to_name[beginning_blocks]} requires d_model == {pretrained_d_model}"

        name = block_size_to_name[beginning_blocks]
        url = f"http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/{name}-PT.tar.gz"
        return load_remote_model(url)

    def load_pretrained_weights(self):
        model_path = self.path_to_pretrained_checkpoint()

        # Our parameter names will look like this: 'blocks.0.layers.2.attention.v_head.bias', but the training
        # files will have the form 'attn_layers.2.v_head.bias'. We need to convert here.
        state_dict = torch.load(str(model_path))
        noninitialized_keys = []

        # Don't forget about the embeddings
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
                    old_weights = old_weights.reshape(*param.data.shape)

                param.data = old_weights
            except KeyError:
                noninitialized_keys.append({'new_name': var_name, 'old_name': old_name})

        if len(noninitialized_keys) > 0:
            logger = logging.getLogger(__name__)
            logger.warning(f'PretrainedModelManager: Failed to initialize weights: {noninitialized_keys}')

    # For the "args" parameter in the old FunnelTFM.__init__()
    def get_backward_compatible_args(self) -> AttributeDict:
        return AttributeDict(  # can use DynamicDict.my_key syntax
            pad_id=self.pad_id,
            num_class=self.num_classes,
            seg_id_cls=self.seg_id_cls,
            truncate_seq=self.truncate_seq,
            attn_type=self.attention_type
        )

    # Get a dictionary compatible with the old ModelConfig class from Funnel-Transformers
    def get_backward_compatible_dict(self) -> Dict:
        return {
            "vocab_size": self.vocab_size,
            "d_embed": self.d_model,
            "d_model": self.d_model,
            "n_head": self.num_heads,
            "d_head": self.d_model // self.num_heads,
            "d_inner": self.d_model * 4,
            "dropout": self.dropout,
            "dropatt": self.attention_dropout,
            "dropact": self.ffn_dropout,
            "block_size": '_'.join([str(x) for x in self.block_sizes]),
            "pooling_type": self.pooling_type,

            # We lose info here since Funnel-Transformers doesn't support different scaling factors for each block
            "pooling_size": self.scaling_factors[0],
            "separate_cls": self.separate_cls,
            "pool_q_only": self.pool_q_only
        }


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
        with attention_state.pooled_q_unpooled_k():
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
