from __future__ import annotations
import torch.nn as nn

from AttentionState import *
from Utilities import *
from ops import LayerNorm
from ops import EmbeddingLookup
from ops import RelativePositionalAttention
from ops import PositionwiseFFN
from ops import Dense


@dataclass
class FunnelConfig(SerializableObject):
    block_sizes: Tuple[int, ...]
    d_model: int
    n_head: int

    # **Default values taken from pretrained config files & flags- do not change**
    vocab_size: int = 30522
    attention_dropout: float = 0.1
    dropout: float = 0.1
    ffn_dropout: float = 0.0
    pooling_type: str = 'mean'
    scaling_factors: Union[int, Tuple[int, ...]] = 2
    separate_cls: bool = True
    pool_q_only: bool = True
    truncate_seq: bool = True
    seg_id_cls: int = 2  # Segment ID of the [CLS] token
    pad_id: int = None
    num_classes: int = 0
    use_classification_head: bool = False

    # Fine to change these
    attention_type: str = 'rel_shift'
    rezero_blocks: Optional[Iterable[int]] = field(default_factory=tuple)  # Blocks for which to use ReZero
    max_position_embeddings: int = 512
    return_attention_state: bool = False    # Useful for a VAE encoder; can reuse the state in the decoder
    use_performer_attention: bool = False
    upsampling: bool = False                 # True for the "reverse" funnel transformer; e.g. a VAE decoder

    def __post_init__(self):
        # Turn a single floating point scaling factor x into (x, x, x...) of the appropriate length
        if isinstance(self.scaling_factors, float):
            factor = self.scaling_factors
            self.scaling_factors = (factor for _ in range(len(self.block_sizes) - 1))

        # Make it so scaling_factors and block_sizes are equal length; last scaling factor is 1 (no scaling)
        if len(self.scaling_factors) < len(self.block_sizes):
            self.scaling_factors += (1,)


class FunnelTransformer(nn.Module):
    def __init__(self, config: FunnelConfig):
        super(FunnelTransformer, self).__init__()
        self.config = config
        self.input_layer = nn.Sequential(
            EmbeddingLookup(config.vocab_size, config.d_model),
            LayerNorm(config.d_model),
            nn.Dropout(config.dropout))

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

    # Add a Callable (e.g. a Module) that will be called with the final hidden state of the block at
    # block_index, just before pooling occurs. A list with the output of all hidden state listeners
    # will be included in the output of forward().
    def add_hidden_state_listener(self, listener: Callable, block_index: int):
        assert not self.config.upsampling
        self.blocks[block_index].add_hidden_state_listener(listener)

    # Add a Callable (e.g. a Module) that will be called with the hidden state of the model at a location
    # (either a block index or a (block index, layer index) tuple)— just AFTER nearest-neighbor upsampling
    # occurs if applicable. The output of *this transform* will then be passed on to the next module. This
    # makes it possible to add extra information (possibly stochastically) to the upsampling process.
    def add_upsampling_transform(self, transform: Callable, location: Union[int, Tuple[int, int]]):
        assert self.config.upsampling

        if isinstance(location, tuple):
            block_index = location[0]
            layer_index = location[1]
        else:
            block_index = location
            layer_index = -1

        self.blocks[block_index].add_upsampling_transform(transform, layer_index)

    # Returns a copy of the transformer whose upsampling parameter is flipped
    def inverted_copy(self, reverse_layer_order: bool = True, reinitialize_blocks = ()):
        new_funnel: FunnelTransformer = self.deepcopy()
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

    # Convenient method for loading old checkpoints
    def iterate_parameters_by_layer(self) -> Iterator[Tuple[str, torch.nn.Parameter, int]]:
        absolute_index = 0
        for block in self.blocks:
            for layer in block.layers:
                for var_name, param in layer.named_parameters():
                    yield var_name, param, absolute_index

                absolute_index += 1

    # All inputs should be of shape (batch, length)
    def forward(self, x: Tensor, input_mask: Tensor = None, seg_id: Tensor = None, cls_target: Tensor = None):
        attn_state = self.attention_state
        attn_state.configure_for_input(x, input_mask, seg_id)

        listener_outputs = []
        x = self.input_layer(x)  # x.shape == (batch, length, d_model)

        q = kv = x
        for block in self.blocks:
            q, kv, temp = block(q, kv, attn_state)
            listener_outputs.extend(temp)

        output = (q,)

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

        if len(listener_outputs):
            output += (listener_outputs,)

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
        self.output_transforms = []

    def add_output_transform(self, transform: Callable):
        self.output_transforms.append(transform)

    # Q is different from K and V right after pooling; K and V are always the same
    def forward(self, q, kv, attention_state: AttentionState):
        # These custom attention and feedforward layers have built-in residual connections
        x = self.attention(q, kv, kv, attention_state)
        x = self.feedforward(x)
        for transform in self.output_transforms:
            x = transform(x)

        return x

class FunnelBlock(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int):
        super().__init__()

        # We specifically don't make this an nn.ModuleList because 1) listeners don't have to be Modules,
        # and 2) listeners that are Modules should be children of the Module that called add_hidden_state_listener
        self.hidden_state_listeners = []

        block_size = config.block_sizes[block_index]
        self.layers = nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])

        self.rezero_alpha = None

    def activate_rezero(self):
        self.rezero_alpha = nn.Parameter(torch.tensor(0))

    def add_hidden_state_listener(self, listener: Callable):
        self.hidden_state_listeners.append(listener)

    def add_upsampling_transform(self, transform: Callable, layer_index: int = -1):
        self.layers[layer_index].add_output_transform(transform)

    def forward(self, q, kv, attention_state: AttentionState) -> Tuple[Tensor, Tensor, List[Any]]:
        with attention_state.pooled_q_unpooled_k():
            kv = self.layers[0](q, kv, attention_state)

        for layer in self.layers[1:]:
            kv = layer(kv, kv, attention_state)

        listener_output = [listener(kv) for listener in self.hidden_state_listeners]

        # With ReZero, we introduce an additional residual connection between blocks, where the output of each
        # block is multiplied by a parameter alpha that is initialized to zero. When alpha == 0, the block has
        # 'no effect' and simply outputs an average pooled version of its input.
        if self.rezero_alpha is not None:
            kv = q + (kv * self.rezero_alpha)

        q = attention_state.scale_input(kv)
        return q, kv, listener_output
