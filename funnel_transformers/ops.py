from __future__ import annotations

import numpy as np
import torch.nn.init as init
from torch import nn
from .AttentionState import *
from ..performers import PerformerAttention, PerformerAttentionConfig

INF = 1e6

try:
    import apex

    LayerNorm = apex.normalization.FusedLayerNorm
except ImportError as e:
    class LayerNorm(nn.LayerNorm):
        def __init__(self, *args, **kwargs):
            super(LayerNorm, self).__init__(*args, **kwargs)
            self.eps = 1e-9

        def forward(self, inputs):
            dtype = torch.float32
            if self.elementwise_affine:
                weight = self.weight.type(dtype)
                bias = self.bias.type(dtype)
            else:
                weight = self.weight
                bias = self.bias
            input_dtype = inputs.dtype
            inputs = inputs.type(dtype)
            output = F.layer_norm(inputs, self.normalized_shape, weight, bias, self.eps)
            if output.dtype != input_dtype:
                output = output.type(input_dtype)
            return output


# All these custom module classes are probably unnecessary, but we keep them them here in order
# to maintain full backward compatibility with the original pretrained Funnel Transformer checkpoints.
class GELU(nn.Module):
    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
        return x * cdf


class EmbeddingLookup(nn.Module):
    def __init__(self, n_embed, d_embed):
        super(EmbeddingLookup, self).__init__()
        self.lookup_table = nn.Parameter(torch.zeros([n_embed, d_embed]))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.lookup_table)

    def forward(self, inputs):
        return F.embedding(inputs, self.lookup_table)


def maybe_convert_to_list(x):
    if isinstance(x, (int, float)):
        return [x]
    elif isinstance(x, (list, tuple)):
        return list(x)


def get_einsum_string(ndims, einsum_symbols=None):
    if einsum_symbols is None:
        einsum_symbols = ["u", "v", "w", "x", "y", "z"]
    assert ndims <= len(einsum_symbols)
    einsum_prefix = ""
    for i in range(ndims):
        einsum_prefix += einsum_symbols[i]

    return einsum_prefix


class Dense(nn.Module):
    def __init__(self, inp_shape, out_shape, bias=True, reverse_order=False):
        super(Dense, self).__init__()

        self.inp_shape = maybe_convert_to_list(inp_shape)
        self.out_shape = maybe_convert_to_list(out_shape)

        self.reverse_order = reverse_order
        if self.reverse_order:
            self.einsum_str = "...{0},{1}{0}->...{1}".format(
                get_einsum_string(len(self.inp_shape), ["a", "b", "c", "d"]),
                get_einsum_string(len(self.out_shape), ["e", "f", "g", "h"]))
            weight_shape = self.out_shape + self.inp_shape
        else:
            self.einsum_str = "...{0},{0}{1}->...{1}".format(
                get_einsum_string(len(self.inp_shape), ["a", "b", "c", "d"]),
                get_einsum_string(len(self.out_shape), ["e", "f", "g", "h"]))
            weight_shape = self.inp_shape + self.out_shape

        self.weight = nn.Parameter(torch.zeros(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_shape))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = np.prod(self.inp_shape)
        fan_out = np.prod(self.out_shape)
        std = np.sqrt(1.0 / float(fan_in + fan_out))

        nn.init.normal_(self.weight, std=std)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, inputs):
        output = torch.einsum(self.einsum_str, inputs, self.weight)
        if self.bias is not None:
            output = output + self.bias

        return output


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_inner, dropout, dropact):
        super(PositionwiseFFN, self).__init__()
        self.pffn = nn.Sequential(
            Dense(d_model, d_inner),
            GELU(),
            nn.Dropout(dropact),
            Dense(d_inner, d_model),
            nn.Dropout(dropout))
        self.layer_norm = LayerNorm(d_model)

    def forward(self, inputs):
        pffn_out = self.pffn(inputs)
        output = self.layer_norm(inputs + pffn_out)  # Residual connection
        return output


class RelativePositionalAttention(nn.Module):
    def __init__(self, net_config):
        super(RelativePositionalAttention, self).__init__()

        d_model, n_head = net_config.d_model, net_config.num_heads
        d_head = d_model // n_head

        self.net_config = net_config
        self.attn_type = net_config.attention_type

        self.dropout = net_config.dropout
        self.dropatt = net_config.attention_dropout

        self.att_drop = nn.Dropout(self.dropatt)
        self.hid_drop = nn.Dropout(self.dropout)

        self.q_head = Dense(d_model, [n_head, d_head], bias=False)
        self.k_head = Dense(d_model, [n_head, d_head])
        self.v_head = Dense(d_model, [n_head, d_head])

        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.r_r_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.r_kernel = nn.Parameter(torch.zeros([d_model, n_head, d_head]))
        self.r_s_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.seg_embed = nn.Parameter(torch.zeros([2, n_head, d_head]))

        self.post_proj = Dense([n_head, d_head], d_model)
        self.layer_norm = LayerNorm(d_model)
        self.normalizer = 1. / np.sqrt(d_head)
        self.reset_parameters()

        if net_config.use_performer_attention:
            performer_config = PerformerAttentionConfig(d_model=d_model, num_heads=n_head, use_linear_layers=False)

            self.content_performer_attention = PerformerAttention(performer_config)
            self.position_performer_attention1 = PerformerAttention(performer_config)
            self.position_performer_attention2 = PerformerAttention(performer_config)

    def reset_parameters(self):
        nn.init.uniform_(self.r_w_bias, b=0.1)
        nn.init.uniform_(self.r_r_bias, b=0.1)
        nn.init.uniform_(self.r_kernel, b=0.1)
        nn.init.uniform_(self.r_s_bias, b=0.1)
        nn.init.uniform_(self.seg_embed, b=0.1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_state: AttentionState) -> Tensor:
        q_head = self.q_head(q)
        k_head = self.k_head(k)
        v_head = self.v_head(v)

        q_head = q_head * self.normalizer
        r_w_bias = self.r_w_bias * self.normalizer
        # content based attention score
        content_score = torch.einsum("...ind,...jnd->...nij", q_head + r_w_bias, k_head)

        pos_bias = self._rel_pos_bias(q_head, k_head.size(1), attn_state)
        seg_bias = self._rel_seg_bias(attn_state.segment_mask, q_head, attn_state)

        # merge attention scores
        attn_score = content_score + pos_bias + seg_bias

        # precision safe
        dtype = attn_score.dtype
        attn_score = attn_score.float()

        # perform masking
        attn_mask = attn_state.attention_mask
        if attn_mask is not None:
            attn_score = attn_score - INF * attn_mask.float()

        # attention distribution
        attn_dist = torch.softmax(attn_score, dim=-1)
        attn_dist = attn_dist.type(dtype)

        attn_dist = self.att_drop(attn_dist)
        # attention output
        attn_vec = torch.einsum("...nij,...jnd->...ind", attn_dist, v_head)

        attn_out = self.post_proj(attn_vec)
        attn_out = self.hid_drop(attn_out)

        output = self.layer_norm(q + attn_out)  # Residual connection
        return output

    @staticmethod
    def _rel_shift(x, row_axis, key_len, shift=1):
        """Perform relative shift to form the relative attention score."""
        # Deal with negative indexing
        row_axis = row_axis % x.ndim

        # Assume `col_axis` = `row_axis + 1`
        col_axis = row_axis + 1
        assert col_axis < x.ndim

        tgt_shape_1, tgt_shape_2 = [], []
        for i in range(x.ndim):
            if i == row_axis:
                tgt_shape_1.append(x.shape[col_axis])
                tgt_shape_2.append(x.shape[row_axis])
            elif i == col_axis:
                tgt_shape_1.append(x.shape[row_axis])
                tgt_shape_2.append(x.shape[col_axis] - shift)
            else:
                tgt_shape_1.append(x.shape[i])
                tgt_shape_2.append(x.shape[i])
        
        y = torch.reshape(x, tgt_shape_1)
        y = torch.narrow(y, row_axis, shift, x.shape[col_axis] - shift)
        y = torch.reshape(y, tgt_shape_2)
        y = torch.narrow(y, col_axis, 0, key_len)

        return y

    def _rel_pos_bias(self, q_head, k_len, attn_state: AttentionState):
        normalizer = self.normalizer
        r_r_bias = self.r_r_bias
        r_kernel = self.r_kernel
        pos_enc = attn_state.positional_encoding

        if self.attn_type == "factorized":
            enc_q_1, enc_q_2, enc_k_1, enc_k_2 = pos_enc
            q_head_r = torch.einsum("...inh,dnh->...ind", q_head + r_r_bias * normalizer, r_kernel)
            q_head_r_1 = q_head_r * torch.unsqueeze(enc_q_1, -2)
            q_head_r_2 = q_head_r * torch.unsqueeze(enc_q_2, -2)
            prefix_k = get_einsum_string(len(enc_k_1.shape) - 2)
            einsum_str = "...ind,{0}jd->...nij".format(prefix_k)

            bd = (torch.einsum(einsum_str, q_head_r_1, enc_k_1) +
                  torch.einsum(einsum_str, q_head_r_2, enc_k_2))
        elif self.attn_type == "rel_shift":
            shift = 1 + attn_state.pooled_q_unpooled_k_flag

            q_head = q_head + r_r_bias * normalizer
            r_head = torch.einsum("td,dnh->tnh", pos_enc, r_kernel)
            bd = torch.einsum("bfnh,tnh->bnft", q_head, r_head)
            bd = self._rel_shift(bd, -2, k_len, shift)
        else:
            raise NotImplementedError

        if attn_state.not_cls_mask is not None:
            bd = bd * attn_state.not_cls_mask
        return bd

    def _rel_seg_bias(self, seg_mat, q_head, attn_state: AttentionState):
        # segment based attention score
        if seg_mat is None:
            seg_bias = 0
        else:
            r_s_bias = self.r_s_bias * self.normalizer
            seg_embed = self.seg_embed

            seg_bias = torch.einsum("...ind,snd->...nis", q_head + r_s_bias, seg_embed)
            tgt_shape = list(seg_mat.size())
            tgt_shape.insert(-2, self.net_config.num_heads)
            seg_mat = torch.unsqueeze(seg_mat, -3).expand(tgt_shape)
            _diff, _same = torch.split(seg_bias, 1, dim=-1)
            _diff = _diff.expand(tgt_shape)
            _same = _same.expand(tgt_shape)
            seg_bias = torch.where(seg_mat, _same, _diff)
            if attn_state.not_cls_mask is not None:
                seg_bias *= attn_state.not_cls_mask

        return seg_bias
