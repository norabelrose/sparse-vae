from dataclasses import dataclass
from torch import nn
from .AttentionState import *
from ..core.Transformers import positional_encodings_like
import numpy as np

BIG_CONST = 1e6


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
    def __init__(self, d_model, d_inner, dropout, dropact, layer_norm_eps=1e-9):
        super(PositionwiseFFN, self).__init__()
        self.pffn = nn.Sequential(
            EinsumLayer("...a,ab->...b", [d_model], [d_inner]),
            GELU(),
            nn.Dropout(dropact),
            EinsumLayer("...b,ba->...a", [d_inner], [d_model]),
            nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, inputs):
        pffn_out = self.pffn(inputs)
        output = self.layer_norm(inputs + pffn_out)  # Residual connection
        return output

@dataclass
class AttentionHparams:
    d_model: int = 768
    num_heads: int = 12

    attention_dropout: float = 0.1
    dropout: float = 0.1
    ffn_dropout: float = 0.0
    layer_norm_eps: float = 1e-9
    separate_cls: bool = False
    use_segment_attention: bool = False

    inject_absolute_pos_encodings: bool = False
    positional_attention_type: str = 'rel_shift'  # 'none', 'rel_shift' or 'factorized'


class FunnelAttention(nn.Module):
    def __init__(self, hparams):
        super(FunnelAttention, self).__init__()

        d_model = hparams.d_model
        n_head = hparams.num_heads
        d_head = d_model // n_head

        self.hparams = hparams
        self.causal = False
        self.separate_cls = hparams.separate_cls

        self.dropout = hparams.dropout
        self.dropatt = hparams.attention_dropout

        self.att_drop = nn.Dropout(self.dropatt)
        self.hid_drop = nn.Dropout(self.dropout)

        # The asymmetry (Q proj has no bias, but K and V do) is for backward comp. with Funnel-Transformers checkpoints
        self.q_head = EinsumLayer("...d,dnh->...nh", [d_model], [n_head, d_head], bias=False)
        self.k_head = EinsumLayer("...d,dnh->...nh", [d_model], [n_head, d_head])
        self.v_head = EinsumLayer("...d,dnh->...nh", [d_model], [n_head, d_head])

        # These parameters are applied *headwise*, hence they have a extra head dimension
        if hparams.positional_attention_type != 'none':
            self.r_w_bias = nn.Parameter(torch.zeros(n_head, 1, d_head))
            self.r_r_bias = nn.Parameter(torch.zeros(n_head, 1, d_head))
            self.r_kernel = nn.Parameter(torch.zeros(n_head, d_model, d_head))
        else:
            self.r_w_bias = None
            self.r_r_bias = None
            self.r_kernel = None

        if hparams.use_segment_attention:
            self.r_s_bias = nn.Parameter(torch.zeros(n_head, d_head))
            self.seg_embed = nn.Parameter(torch.zeros(2, n_head, d_head))
        else:
            self.r_s_bias = None
            self.seg_embed = None

        self.post_proj = EinsumLayer("...lnh,nhd->...ld", [n_head, d_head], [d_model])
        self.layer_norm = nn.LayerNorm(d_model, eps=hparams.layer_norm_eps)
        self.normalizer = 1. / np.sqrt(d_head)
        self.reset_parameters()

    def reset_parameters(self):
        if self.r_w_bias is not None:
            nn.init.uniform_(self.r_w_bias, b=0.1)
            nn.init.uniform_(self.r_r_bias, b=0.1)
            nn.init.uniform_(self.r_kernel, b=0.1)
        if self.r_s_bias is not None:
            nn.init.uniform_(self.r_s_bias, b=0.1)
            nn.init.uniform_(self.seg_embed, b=0.1)
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_state: AttentionState) -> Tensor:
        input_q = q  # Save for residual connection

        # Inject absolute positional encodings at each layer if we're not using relative positional attention
        if self.hparams.inject_absolute_pos_encodings:
            q, k, v = (x + positional_encodings_like(x) for x in (q, k, v))

        q = self.q_head(q)
        k = self.k_head(k)
        v = self.v_head(v)

        q, k, v = (rearrange(x, '... l h d -> ... h l d') for x in (q, k, v))

        if self.r_w_bias is not None:
            attn_score = (q + self.r_w_bias) @ k.transpose(-2, -1) * self.normalizer
            attn_score = attn_score + self._attn_pos_term(q, k.size(-2), attn_state)
        else:
            attn_score = q @ k.transpose(-2, -1) * self.normalizer

        if (not_cls := attn_state.get_not_cls_mask()) is not None:
            attn_score *= not_cls

        if self.r_s_bias is not None:
            attn_score = attn_score + self._attn_seg_term(q, attn_state)

        # Perform masking
        attn_mask = attn_state.get_attention_mask()
        if self.causal:
            causal_mask = torch.ones_like(attn_score)
            causal_mask.triu_(1)
            attn_mask = causal_mask if attn_mask is None else attn_mask * causal_mask

        if attn_mask is not None:
            attn_score = attn_score - BIG_CONST * attn_mask

        # Attention distribution
        attn_dist = torch.softmax(attn_score, dim=-1)
        attn_dist = self.att_drop(attn_dist)
        output = attn_dist @ v
        
        output = rearrange(output, "... h l d -> ... l h d")

        # attention output
        attn_out = self.post_proj(output)
        attn_out = self.hid_drop(attn_out)

        output = self.layer_norm(input_q + attn_out)  # Residual connection
        return output

    # A^{position} in the paper
    def _attn_pos_term(self, q, k_len, attn_state: AttentionState):
        pos_enc = attn_state.get_positional_encodings()
        attn_type = self.hparams.positional_attention_type
        
        if attn_type == "factorized":
            enc_q_1, enc_q_2, enc_k_1, enc_k_2 = pos_enc  # seq_len, d_model
            q_r = (q + self.r_r_bias) @ self.r_kernel.transpose(-2, -1) * self.normalizer

            # Broadcast positional encodings across the batch and head dimensions
            q_len = q.shape[-2]
            q_r_1 = q_r * enc_q_1[:q_len]
            q_r_2 = q_r * enc_q_2[:q_len]

            scores = q_r_1 @ enc_k_1[:k_len].transpose(-2, -1) + q_r_2 @ enc_k_2[:k_len].transpose(-2, -1)

        elif attn_type == "rel_shift":
            shift = 1 + (attn_state.block_begin_flag and attn_state.current_block > 0)
            # shift = 0

            # Funnel Transformer paper, page 13
            scores = (q + self.r_r_bias) @ (pos_enc @ self.r_kernel).transpose(-2, -1) * self.normalizer

            # Do the "relative shift"
            target_shape1 = list(scores.shape)
            target_shape2 = target_shape1.copy()
            target_shape1[-2], target_shape1[-1] = target_shape1[-1], target_shape1[-2]
            
            scores = scores.reshape(target_shape1)
            target_shape2[-1] -= shift
            scores = scores.narrow(-2, shift, target_shape2[-1])
            scores = scores.reshape(target_shape2)
            scores = scores.narrow(-1, 0, k_len)
        else:
            raise NotImplementedError

        return scores

    def _attn_seg_term(self, q_head, attn_state: AttentionState):
        # segment based attention score
        segment_matrix = attn_state.get_segment_matrix()
        if segment_matrix is None:
            seg_term = 0
        else:
            r_s_bias = self.r_s_bias.unsqueeze(-2) * self.normalizer

            seg_term = torch.einsum("...ind,snd->...nis", (q_head + r_s_bias).movedim(2, 1), self.seg_embed)
            tgt_shape = list(segment_matrix.shape)
            tgt_shape.insert(-2, self.hparams.num_heads)
            segment_matrix = segment_matrix.unsqueeze(-3).expand(tgt_shape)
            different, same = seg_term.chunk(2, dim=-1)
            different = different.expand(tgt_shape)
            same = same.expand(tgt_shape)
            seg_term = torch.where(segment_matrix, same, different)

        return seg_term
