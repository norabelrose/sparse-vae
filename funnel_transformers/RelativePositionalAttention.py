from __future__ import annotations

import numpy as np
from einops import rearrange
from torch import nn
from .AttentionState import *
from ..performers import PerformerAttentionConfig, PerformerAttention

BIG_CONST = 1e6

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


class EinsumLayer(nn.Module):
    def __init__(self, einsum_formula: str, input_shape: List[int], output_shape: List[int], bias: bool=True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(input_shape + output_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_shape))
        else:
            self.register_parameter("bias", None)
        
        self.einsum_formula = einsum_formula
        
        # Initialize weights
        fan_in = np.prod(input_shape)
        fan_out = np.prod(output_shape)
        std = np.sqrt(1.0 / float(fan_in + fan_out))

        nn.init.normal_(self.weight, std=std)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)
    
    def forward(self, inputs):
        output = torch.einsum(self.einsum_formula, inputs, self.weight)
        if self.bias is not None:
            output = output + self.bias
        
        return output


# These custom module classes are probably unnecessary, but we keep them them here in order
# to maintain full backward compatibility with the original pretrained Funnel Transformer checkpoints.
class GELU(nn.Module):
    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
        return x * cdf


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_inner, dropout, dropact):
        super(PositionwiseFFN, self).__init__()
        self.pffn = nn.Sequential(
            EinsumLayer("...a,ab->...b", [d_model], [d_inner]),
            GELU(),
            nn.Dropout(dropact),
            EinsumLayer("...b,ba->...a", [d_inner], [d_model]),
            nn.Dropout(dropout))
        self.layer_norm = LayerNorm(d_model)

    def forward(self, inputs):
        pffn_out = self.pffn(inputs)
        output = self.layer_norm(inputs + pffn_out)  # Residual connection
        return output


class RelativePositionalAttention(nn.Module):
    def __init__(self, hparams):
        super(RelativePositionalAttention, self).__init__()

        d_model, n_head = hparams.d_model, hparams.num_heads
        d_head = d_model // n_head

        self.hparams = hparams
        self.attn_type = hparams.attention_type

        self.dropout = hparams.dropout
        self.dropatt = hparams.attention_dropout

        self.att_drop = nn.Dropout(self.dropatt)
        self.hid_drop = nn.Dropout(self.dropout)

        # The asymmetry (Q proj has no bias, but K and V do) is for backward comp. with Funnel-Transformers checkpoints
        self.q_head = EinsumLayer("...d,dnh->...nh", [d_model], [n_head, d_head], bias=False)
        self.k_head = EinsumLayer("...d,dnh->...nh", [d_model], [n_head, d_head])
        self.v_head = EinsumLayer("...d,dnh->...nh", [d_model], [n_head, d_head])

        # These parameters are applied *headwise*, hence they have a extra head dimension
        self.r_w_bias = nn.Parameter(torch.zeros(n_head, 1, d_head))
        self.r_r_bias = nn.Parameter(torch.zeros(n_head, 1, d_head))
        self.r_kernel = nn.Parameter(torch.zeros(n_head, d_model, d_head))
        self.r_s_bias = nn.Parameter(torch.zeros(n_head, d_head))
        self.seg_embed = nn.Parameter(torch.zeros(2, n_head, d_head))

        self.post_proj = EinsumLayer("blnh,nhd->bld", [n_head, d_head], [d_model])
        self.layer_norm = LayerNorm(d_model)
        self.normalizer = 1. / np.sqrt(d_head)
        self.reset_parameters()

        if hparams.use_performer_attention:
            assert hparams.attention_type == "factorized",\
                "Performer attention is only compatible with the factorized form of relative positional attention"

            # One module for content attention, and two for positional attention
            performer_config = PerformerAttentionConfig(d_model=d_model, num_heads=n_head, use_linear_layers=False)
            self.performer_attentions = [PerformerAttention(performer_config) for _ in range(3)]

    def reset_parameters(self):
        nn.init.uniform_(self.r_w_bias, b=0.1)
        nn.init.uniform_(self.r_r_bias, b=0.1)
        nn.init.uniform_(self.r_kernel, b=0.1)
        nn.init.uniform_(self.r_s_bias, b=0.1)
        nn.init.uniform_(self.seg_embed, b=0.1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_state: AttentionState) -> Tensor:
        input_q = q  # Save for residual connection
        q = self.q_head(q)
        k = self.k_head(k)
        v = self.v_head(v)

        q, k, v = (rearrange(x, 'b l h d -> b h l d') for x in (q, k, v))

        if self.hparams.use_performer_attention:
            # "Funnel Transformers" page 13, formula 8 and page 14, final formula of section A.2
            r_r = self.r_r_bias
            q_i = (q + r_r) @ self.r_kernel.transpose(-2, -1)

            # See the trigonometric identities in the paper on page 14
            phi_i, pi_i, psi_j, omega_j = attn_state.positional_encoding

            # This is my own derivation. Basically, the exp(A^{content} + A^{position}) in relative positional attention
            # can be decomposed into the elementwise product exp(A^{content}) * exp(A^{position}), and with the
            # factorized formulation, exp(A^{position}) can be decomposed into another product of exponentials. We then
            # have the product of three exponentials of dot products, and each can be approximated with a Performer
            # attention kernel function with O(n) time and space complexity. In order to have an unbiased estimate,
            # though, we need to draw the random features for each term independently, hence the three separate
            # PerformerAttention objects.
            pos_query1, pos_query2, pos_key1, pos_key2 = q_i * phi_i, psi_j, q_i * pi_i, omega_j
            qk_pairs = [(q, k), (pos_query1, pos_key1), (pos_query2, pos_query2)]
            norm = self.normalizer ** 0.5   # We multiply both Q and K by the 4th root of d_model, not Q by its sqrt

            # Get the elementwise products of all the Q', K', and denominators
            q_prime_prod, k_prime_prod, denom_prod = norm, norm, 1.0
            for attn, (query, key) in zip(self.performer_attentions, qk_pairs):
                q_prime, k_prime = attn.get_projected_queries_and_keys(query, key)
                denom = attn.denominator_for_projected_queries_and_keys(query, key)

                q_prime_prod *= q_prime; k_prime_prod *= k_prime; denom_prod *= denom

            k_prime_prod = self.att_drop(k_prime_prod)
            output = q_prime_prod @ (k_prime_prod @ v) / denom_prod

        else:
            # content based attention score
            r_w = self.r_w_bias[:, None, :]
            content_score = (q + r_w) @ k.transpose(-2, -1) * self.normalizer
            
            pos_bias = self._attn_pos_term(q, k.size(-2), attn_state)
            seg_bias = self._attn_seg_term(attn_state.segment_mask, q, attn_state)

            # merge attention scores
            attn_score = content_score + pos_bias + seg_bias

            # precision safe
            dtype = attn_score.dtype
            attn_score = attn_score.float()

            # perform masking
            attn_mask = attn_state.attention_mask
            if attn_mask is not None:
                attn_score = attn_score - BIG_CONST * attn_mask.float()

            # attention distribution
            attn_dist = torch.softmax(attn_score, dim=-1)
            attn_dist = attn_dist.type(dtype)
            attn_dist = self.att_drop(attn_dist)
            output = attn_dist @ v
        
        output = rearrange(output, "b h l d -> b l h d")

        # attention output
        attn_out = self.post_proj(output)
        attn_out = self.hid_drop(attn_out)

        output = self.layer_norm(input_q + attn_out)  # Residual connection
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

    # A^{position} in the paper
    def _attn_pos_term(self, q, k_len, attn_state: AttentionState):
        pos_enc = attn_state.positional_encoding
        r_r = self.r_r_bias[:, None, :]

        if self.attn_type == "factorized":
            enc_q_1, enc_q_2, enc_k_1, enc_k_2 = pos_enc    # seq_len, d_model
            q_r = (q + r_r) @ self.r_kernel.transpose(-2, -1) * self.normalizer

            # Broadcast positional encodings across the batch and head dimensions
            q_r_1 = q_r * enc_q_1
            q_r_2 = q_r * enc_q_2

            bd = q_r_1 @ enc_k_1.transpose(-2, -1) + q_r_2 @ enc_k_2.transpose(-2, -1)

        elif self.attn_type == "rel_shift":
            shift = 1 + attn_state.block_begin_flag

            # Funnel Transformer paper, page 13
            bd = (q + r_r) @ (pos_enc @ self.r_kernel).transpose(-2, -1) * self.normalizer
            bd = self._rel_shift(bd, -2, k_len, shift)
        else:
            raise NotImplementedError

        if attn_state.not_cls_mask is not None:
            bd *= attn_state.not_cls_mask
        
        return bd

    def _attn_seg_term(self, seg_mat, q_head, attn_state: AttentionState):
        # segment based attention score
        if seg_mat is None:
            seg_bias = 0
        else:
            r_s_bias = self.r_s_bias * self.normalizer
            seg_embed = self.seg_embed

            seg_bias = torch.einsum("...ind,snd->...nis", q_head + r_s_bias, seg_embed)
            tgt_shape = list(seg_mat.size())
            tgt_shape.insert(-2, self.hparams.num_heads)
            seg_mat = torch.unsqueeze(seg_mat, -3).expand(tgt_shape)
            _diff, _same = torch.split(seg_bias, 1, dim=-1)
            _diff = _diff.expand(tgt_shape)
            _same = _same.expand(tgt_shape)
            seg_bias = torch.where(seg_mat, _same, _diff)
            if attn_state.not_cls_mask is not None:
                seg_bias *= attn_state.not_cls_mask

        return seg_bias
