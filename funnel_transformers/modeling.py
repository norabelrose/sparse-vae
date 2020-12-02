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


def parse_depth_string(depth_str):
    depth_config = depth_str.split("x")
    if len(depth_config) == 1:
        depth_config.append(1)
    assert len(depth_config) == 2, "Require two-element depth config."

    return list(map(int, depth_config))


@dataclass
class FunnelConfig(SerializableObject):
    """ModelConfig contains fixed hyperparameters of a FunnelTFM model."""

    # Maps the 'old' names of attributes used by the pretrained Funnel Transformer config files to the
    # new, more descriptive names that we use. If there's no change, we don't map it here.
    old_to_new: ClassVar[dict] = {
        "dropatt": 'attention_dropout',
        "dropact": 'ffn_dropout'
    }

    vocab_size: int
    d_embed: int
    d_model: int
    n_head: int
    d_head: int
    d_inner: int
    dropout: float
    attention_dropout: float
    ffn_dropout: float
    block_size: Union[str, List[str]]
    pooling_type: str
    pooling_size: int
    separate_cls: bool
    pool_q_only: bool

    attention_type: str = "rel_shift"
    max_position_embeddings: int = 512
    use_performer_attention: bool = False

    def __post_init__(self):
        block_size = self.block_size.split("_")
        self.n_block = len(block_size)
        self.block_rep = []
        self.block_param = []
        for i, _ in enumerate(block_size):
            block_size_i = parse_depth_string(block_size[i])
            self.block_param.append(block_size_i[0])
            self.block_rep.append(block_size_i[1])

    @classmethod
    def from_old_config(cls, old_data: dict) -> FunnelConfig:
        for old_key, value in old_data.items():
            if old_key in cls.old_to_new:
                new_key = cls.old_to_new[old_key]
                old_data[new_key] = old_data.pop(old_key)

        return cls.from_dict(old_data)

    @staticmethod
    def init_from_args(args):
        """Initialize ModelConfig from args."""
        print("Initialize ModelConfig from args.")
        conf_args = {}
        for key in FunnelConfig.keys:
            conf_args[key] = getattr(args, key)

        return FunnelConfig(**conf_args)


class FunnelTransformer(nn.Module):
    """FunnelTFM model."""
    def __init__(self, funnel_config: FunnelConfig, args, cls_target=True):
        super(FunnelTransformer, self).__init__()
        self.funnel_config = funnel_config
        self.args = args
        self.input_layer = nn.Sequential(
            EmbeddingLookup(funnel_config.vocab_size,
                            funnel_config.d_embed),
            LayerNorm(funnel_config.d_embed),
            nn.Dropout(funnel_config.dropout))

        self.pos_drop = nn.Dropout(funnel_config.dropout)

        self.attn_info = AttentionStructure(funnel_config, args)
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        for block_idx in range(funnel_config.n_block):
            for _ in range(funnel_config.block_param[block_idx]):
                self.attn_layers.append(
                    RelativePositionalAttention(
                        funnel_config,
                        args,
                        funnel_config.d_model,
                        funnel_config.n_head,
                        funnel_config.d_head,
                        funnel_config.dropout,
                        funnel_config.attention_dropout,
                        block_idx,
                    )
                )
                self.pffn_layers.append(
                    PositionwiseFFN(
                        funnel_config.d_model,
                        funnel_config.d_inner,
                        funnel_config.dropout,
                        funnel_config.ffn_dropout,
                    )
                )
        if cls_target:
            self.cls_head = nn.Sequential(
                Dense(funnel_config.d_model, funnel_config.d_model),
                nn.Tanh(),
                nn.Dropout(funnel_config.dropout),
                Dense(funnel_config.d_model, self.args.num_class))
            self.cls_loss = nn.CrossEntropyLoss()

    def tfmxl_layers(self, inputs, seg_id=None, input_mask=None):
        net_config = self.funnel_config
        output = inputs

        ret_dict = {}
        ##### TFM-XL layers
        hiddens = []
        layer_idx = 0
        attn_struct = self.attn_info.init_attn_structure(output, seg_id, input_mask)
        for block_idx in range(net_config.n_block):
            if net_config.separate_cls:
                pooling_flag = output.size(1) > 2
            else:
                pooling_flag = output.size(1) > 1

            if block_idx > 0 and pooling_flag:
                pooled_out, attn_struct, _ = self.attn_info.pre_attn_pooling(
                    output, attn_struct)
            for param_idx in range(net_config.block_param[block_idx]):
                for rep_idx in range(net_config.block_rep[block_idx]):
                    sub_idx = param_idx * net_config.block_rep[block_idx] + rep_idx
                    do_pooling = sub_idx == 0 and block_idx > 0 and pooling_flag

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

                    hiddens.append(output)

                layer_idx += 1
        # print(torch.max(hiddens[-1][0][0]))
        return hiddens, ret_dict

    def extract_hiddens(self, inputs, seg_id=None, input_mask=None):
        word_embed = self.input_layer(inputs)
        hiddens, tfm_dict = self.tfmxl_layers(
            word_embed,
            seg_id=seg_id,
            input_mask=input_mask)
        return [word_embed] + hiddens, tfm_dict

    def forward(self, inputs, input_mask=None, seg_id=None, cls_target=None):
        if input_mask is None and self.args.pad_id is not None:
            input_mask = inputs == self.args.pad_id
            input_mask = input_mask.float()

        hiddens, tfm_dict = self.extract_hiddens(
            inputs, seg_id=seg_id, input_mask=input_mask)

        ret_dict = {}
        if cls_target is not None:
            last_hidden = hiddens[-1][:, 0]
            cls_logits = self.cls_head(last_hidden)
            prediction = torch.argmax(cls_logits, -1)
            ret_dict["cls_pred"] = prediction
            cls_loss = self.cls_loss(cls_logits, cls_target)
            ret_dict["cls_loss"] = cls_loss
            cls_correct = prediction == cls_target
            cls_correct = cls_correct.type(torch.float32).sum()
            ret_dict["cls_corr"] = cls_correct
        update_monitor_dict(ret_dict, tfm_dict)
        return hiddens, ret_dict


def update_monitor_dict(tgt, src, prefix=None):
    if prefix is None:
        tgt.update(src)
    else:
        for k, v in src.items():
            tgt["{}/{}".format(prefix, k)] = v

    return tgt
