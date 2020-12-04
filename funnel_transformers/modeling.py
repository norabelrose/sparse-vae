from __future__ import annotations
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import *
from ..Utilities import SerializableObject
from .ops import LayerNorm
from .ops import EmbeddingLookup
from .ops import RelativePositionalAttention
from .ops import PositionwiseFFN
from .ops import Dense
from .ops import AttentionStructure


@dataclass
class FunnelConfig(SerializableObject):
    """Contains fixed hyperparameters of a FunnelTFM model."""

    block_sizes: Tuple[int, ...]
    d_model: int
    n_head: int

    # **Default values taken from pretrained config files & flags- do not change**
    vocab_size: int = 30522
    attention_dropout: float = 0.1
    dropout: float = 0.1
    ffn_dropout: float = 0.0
    pooling_type: str = 'mean'
    scaling_factors: Union[float, Tuple[float, ...]] = 2.0
    separate_cls: bool = True
    pool_q_only: bool = True
    truncate_seq: bool = True
    seg_id_cls: int = 2  # Segment ID of the [CLS] token
    pad_id: int = None
    num_classes: int = 0
    use_classification_head: bool = False

    # Fine to change these
    attention_type: str = 'rel_shift'
    max_position_embeddings: int = 512
    use_performer_attention: bool = False

    # Which hidden states to return on forward(). Possible values: 'all', 'per_block', 'final'.
    hiddens_to_return: str = 'all'

    def __post_init__(self):
        # Turn a single floating point scaling factor x into (x, x, x...) of the appropriate length
        if isinstance(self.scaling_factors, float):
            factor = self.scaling_factors
            self.scaling_factors = (factor for _ in range(len(self.block_sizes) - 1))

        # Make it so scaling_factors and block_sizes are equal length; last scaling factor is 1.0 (no scaling)
        if len(self.scaling_factors) < len(self.block_sizes):
            self.scaling_factors += (1.0,)


class FunnelTransformer(nn.Module):
    """FunnelTFM model."""
    def __init__(self, config: FunnelConfig):
        super(FunnelTransformer, self).__init__()
        self.config = config
        self.input_layer = nn.Sequential(
            EmbeddingLookup(config.vocab_size, config.d_model),
            LayerNorm(config.d_model),
            nn.Dropout(config.dropout))

        self.pos_drop = nn.Dropout(config.dropout)

        self.attn_info = None
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        for block_index, size in enumerate(config.block_sizes):
            for _ in range(size):
                self.attn_layers.append(
                    RelativePositionalAttention(config, block_index)
                )
                self.pffn_layers.append(
                    PositionwiseFFN(
                        config.d_model,
                        config.d_model * 4,
                        config.dropout,
                        config.ffn_dropout,
                    )
                )
        if config.use_classification_head:
            self.cls_head = nn.Sequential(
                Dense(config.d_model, config.d_model),
                nn.Tanh(),
                nn.Dropout(config.dropout),
                Dense(config.d_model, self.config.num_classes))
            self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, input_mask=None, seg_id=None, cls_target=None):
        if input_mask is None and self.config.pad_id is not None:
            input_mask = inputs == self.config.pad_id
            input_mask = input_mask.float()

        word_embed = self.input_layer(inputs)
        net_config = self.config
        output = inputs

        hiddens = []
        if net_config.hiddens_to_return == 'all':
            hiddens.append(word_embed)

        # Lazily create the Attention Structure object, or create a new one if the
        # (padded) sequence length, dtype, or device of our input changed since last time
        attn_info = self.attn_info
        seq_len, dtype, device = output.size(1), output.dtype, output.device
        if not attn_info or (seq_len, dtype, device) != (attn_info.seq_len, attn_info.dtype, attn_info.device):
            self.attn_info = AttentionStructure(net_config, seq_len, dtype, device)

        layer_idx = 0
        attn_struct = attn_info.get_fresh_attn_tuple(seg_id, input_mask)
        for block_idx, size in enumerate(net_config.block_sizes):
            # Just make sure we don't shrink the length down further than 1 token
            pooling_flag = output.size(1) > 2 if net_config.separate_cls else output.size(1) > 1

            if block_idx > 0 and pooling_flag:
                pooled_out, attn_struct, _ = attn_info.pre_attn_pooling(output, attn_struct)
            for param_idx in range(size):
                do_pooling = param_idx == 0 and block_idx > 0 and pooling_flag

                q = k = v = output
                # update q, k, v for the first sub layer of a pooling block
                if do_pooling:
                    if net_config.pool_q_only:
                        q = pooled_out
                        k = v = output
                    else:
                        q = k = v = pooled_out
                output = self.attn_layers[layer_idx](q, k, v, attn_struct)
                output = self.pffn_layers[layer_idx](output)
                if do_pooling:
                    attn_struct = self.attn_info.post_attn_pooling(attn_struct)

                if net_config.hiddens_to_return == 'all':
                    hiddens.append(output)

                layer_idx += 1

            if net_config.hiddens_to_return == 'per_block':
                hiddens.append(output)

        if net_config.hiddens_to_return == 'final':
            hiddens.append(output)

        if cls_target is not None:
            ret_dict = {}

            last_hidden = hiddens[-1][:, 0]
            cls_logits = self.cls_head(last_hidden)
            prediction = torch.argmax(cls_logits, -1)
            ret_dict["cls_pred"] = prediction
            cls_loss = self.cls_loss(cls_logits, cls_target)
            ret_dict["cls_loss"] = cls_loss
            cls_correct = prediction == cls_target
            cls_correct = cls_correct.type(torch.float32).sum()
            ret_dict["cls_corr"] = cls_correct

            return hiddens, ret_dict
        else:
            return hiddens


def update_monitor_dict(tgt, src, prefix=None):
    if prefix is None:
        tgt.update(src)
    else:
        for k, v in src.items():
            tgt["{}/{}".format(prefix, k)] = v

    return tgt


class FunnelLayer(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int):
        super().__init__()

        self.attention = RelativePositionalAttention(config, block_index)
        self.feedforward = PositionwiseFFN(config.d_model, config.d_model * 4, config.dropout, config.ffn_dropout)

    # Q is different from K and V right after pooling; K and V are always the same
    def forward(self, q, kv, attention_struct):
        x = self.attention(q, kv, kv, attention_struct)
        return self.feedforward(x)


class FunnelBlock(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int):
        super().__init__()

        block_size = config.block_sizes[block_index]
        self.layers = [FunnelLayer(config, block_index) for _ in range(block_size)]

        # Don't scale if we're the last block
        self.scale_factor = config.scaling_factors if block_index != len(config.block_sizes) - 1 else 1.0

    def forward(self, q, kv, attention_struct):
        x = self.layers[0](q, kv, attention_struct)

        for layer in self.layers[1:]:
            x = layer(x, attention_struct)

        if self.scale_factor == 1.0:
            return x
