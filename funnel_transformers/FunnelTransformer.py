from __future__ import annotations

from .AttentionState import AttentionState
from .FunnelConfig import FunnelConfig
from ..Utilities import *
from copy import deepcopy
from .ops import LayerNorm
from .ops import EmbeddingLookup
from .ops import RelativePositionalAttention
from .ops import PositionwiseFFN
from .ops import Dense
import torch.nn as nn

class FunnelTransformer(nn.Module):
    def __init__(self, config: FunnelConfig):
        super(FunnelTransformer, self).__init__()
        self.config = config

        if not config.upsampling:
            self.input_layer = nn.Sequential(
                EmbeddingLookup(config.vocab_size, config.d_model),
                LayerNorm(config.d_model),
                nn.Dropout(config.dropout))
        else:
            self.output_linear = nn.Linear(config.d_model, config.vocab_size)

        self.blocks = nn.ModuleList([FunnelBlock(config, index) for index in range(len(config.block_sizes))])
        for block_index in config.rezero_blocks:
            self.blocks[block_index].activate_rezero()

        self.attention_state = AttentionState(config)

        if config.use_classification_head:
            self.cls_head = nn.Sequential(
                Dense(config.d_model, config.d_model),
                nn.Tanh(),
                nn.Dropout(config.dropout),
                Dense(config.d_model, self.config.num_classes))
            self.cls_loss = nn.CrossEntropyLoss()

    # Returns a copy of the transformer whose upsampling parameter is flipped
    def inverted_copy(self, reverse_layer_order: bool = True, reinitialize_blocks = ()):
        new_funnel: FunnelTransformer = deepcopy(self)
        new_funnel.config.upsampling = not self.config.upsampling

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

    # All inputs should be of shape (batch, length)
    def forward(self, x: Tensor, input_mask: Tensor = None, seg_id: Tensor = None, cls_target: Tensor = None):
        hidden_states = []
        if not self.config.upsampling:
            x = self.input_layer(x)  # x.shape == (batch, length, d_model)
        
        attn_state = self.attention_state
        attn_state.configure_for_input(x, input_mask, seg_id)

        q = kv = x
        for block in self.blocks:
            q, kv = block(q, kv, attn_state)

            if self.config.return_block_outputs:
                hidden_states.append(kv)

        # Non-autoregressively generate a softmax distribution over words
        if self.config.upsampling:
            q = self.output_linear(q)
            q = nn.functional.softmax(q, dim=-1)

        # When return_block_outputs == True, we return a list of hidden states (including the last output)
        output = (q,) if not hidden_states else (hidden_states,)

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

            output += (ret_dict,)

        if self.config.return_attention_state:
            attn_state.reset(keep_masks=True)
            output += (attn_state,)
        else:
            attn_state.reset()
        return output

class FunnelLayer(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int):
        super().__init__()

        self.attention = RelativePositionalAttention(config, block_index)
        self.feedforward = PositionwiseFFN(config.d_model, config.d_model * 4, config.dropout, config.ffn_dropout)

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
    def __init__(self, config: FunnelConfig, block_index: int):
        super().__init__()

        block_size = config.block_sizes[block_index]
        self.layers = nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])

        self.rezero_alpha = None

    def activate_rezero(self):
        self.rezero_alpha = nn.Parameter(torch.tensor(0))

    def forward(self, q, kv, attention_state: AttentionState) -> Tuple[Tensor, Tensor, List[Any]]:
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
