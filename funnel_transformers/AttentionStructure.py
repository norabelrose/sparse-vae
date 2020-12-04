from .modeling import *
import torch.nn.functional as F


class AttentionStructure(nn.Module):
    """Relative multi-head attention."""

    def __init__(self, net_config: FunnelConfig, seq_len, dtype=torch.float32, device=None):
        super(AttentionStructure, self).__init__()

        self.net_config = net_config
        self.dtype = dtype
        self.device = device
        self.sin_drop = nn.Dropout(net_config.dropout)
        self.cos_drop = nn.Dropout(net_config.dropout)
        self.attn_type = net_config.attention_type
        self.seq_len = seq_len
        self.delta = None

        # Save these for later
        self.pos_enc = self.get_pos_enc(seq_len, dtype, device)
        if net_config.separate_cls:
            self.func_mask = F.pad(torch.ones([seq_len - 1, seq_len - 1], dtype=dtype, device=device), (1, 0, 1, 0))
        else:
            self.func_mask = None

    def stride_pool_pos(self, pos_id, bidx):
        net_config = self.net_config
        if net_config.separate_cls:
            # Under separate [cls], we treat the [cls] as the first token in
            # the previous block of the 1st real block. Since the 1st real
            # block always has position 1, the position of the previous block
            # will 1 - 2**bidx, where `2 ** bidx` is the current stride.
            cls_pos = pos_id.new_tensor([-2 ** bidx + 1])
            if net_config.truncate_seq:
                pooled_pos_id = pos_id[1:-1]
            else:
                pooled_pos_id = pos_id[1:]
            pooled_pos_id = torch.cat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            pooled_pos_id = pos_id[::2]

        return pooled_pos_id

    def construct_rel_pos_seq(self, q_pos, q_stride, k_pos, k_stride):
        shift = q_stride // k_stride

        ref_point = q_pos[0] - k_pos[0]
        num_remove = shift * len(q_pos)
        max_dist = ref_point + num_remove * k_stride
        min_dist = q_pos[0] - k_pos[-1]

        return torch.arange(max_dist, min_dist - 1, -k_stride, dtype=torch.long, device=q_pos.device)

    def get_pos_enc(self, seq_len, dtype, device):
        """Create inputs related to relative position encoding."""
        if self.attn_type == "factorized":
            pos_seq = torch.arange(0, seq_len, 1.0, dtype=dtype, device=device)
            pos_seq_q, pos_seq_k = pos_seq, pos_seq
            d_model = self.net_config.d_model
            d_model_half = d_model // 2
            freq_seq = torch.arange(0, d_model_half, 1.0,
                                    dtype=dtype, device=device)
            inv_freq = 1 / (10000 ** (freq_seq / d_model_half))
            sinusoid_q = torch.einsum("...i,d->...id", pos_seq_q, inv_freq)
            sinusoid_k = torch.einsum("...i,d->...id", pos_seq_k, inv_freq)
            sin_enc_q = torch.sin(sinusoid_q)
            cos_enc_q = torch.cos(sinusoid_q)
            sin_enc_q = self.sin_drop(sin_enc_q)
            cos_enc_q = self.cos_drop(cos_enc_q)
            sin_enc_k = torch.sin(sinusoid_k)
            cos_enc_k = torch.cos(sinusoid_k)
            enc_q_1 = torch.cat([sin_enc_q, sin_enc_q], dim=-1)
            enc_k_1 = torch.cat([cos_enc_k, sin_enc_k], dim=-1)
            enc_q_2 = torch.cat([cos_enc_q, cos_enc_q], dim=-1)
            enc_k_2 = torch.cat([-sin_enc_k, cos_enc_k], dim=-1)
            return [enc_q_1, enc_q_2, enc_k_1, enc_k_2]
        elif self.attn_type == "rel_shift":
            d_model = self.net_config.d_model
            d_model_half = d_model // 2
            freq_seq = torch.arange(0, d_model_half, 1.0,
                                    dtype=dtype, device=device)
            inv_freq = 1 / (10000 ** (freq_seq / d_model_half))

            # initialize an extra long position sequnece
            rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0,
                                      dtype=dtype, device=device)
            zero_offset = seq_len * 2

            sinusoid = torch.einsum("...i,d->...id", rel_pos_id, inv_freq)
            sin_enc = torch.sin(sinusoid)
            cos_enc = torch.cos(sinusoid)
            sin_enc = self.sin_drop(sin_enc)
            cos_enc = self.cos_drop(cos_enc)
            pos_enc = torch.cat([sin_enc, cos_enc], dim=-1)

            # Pre-compute and cache the rel_pos_id for all blocks
            pos_id = torch.arange(0, seq_len, dtype=dtype, device=device)
            pooled_pos_id = pos_id
            pos_enc_list = []

            config = self.net_config
            q_stride, k_stride = 1.0
            for bidx, scale_factor in enumerate(config.scaling_factors):
                # For each block with bidx > 0, we need two types pos_encs:
                #   - Attn(pooled-q, unpooled-kv)
                #   - Attn(pooled-q, pooled-kv)

                #### First type: Attn(pooled-q, unpooled-kv)
                if bidx > 0:
                    pooled_pos_id = self.stride_pool_pos(pos_id, bidx)

                    # construct rel_pos_id
                    rel_pos_id = self.construct_rel_pos_seq(
                        q_pos=pooled_pos_id, q_stride=q_stride,
                        k_pos=pos_id, k_stride=k_stride)

                    # K gets scaled to the same scale as Q after the first attention layer of the block
                    k_stride = q_stride

                    # gather relative positional encoding
                    rel_pos_id = rel_pos_id[:, None] + zero_offset
                    rel_pos_id = rel_pos_id.expand(rel_pos_id.size(0), d_model)
                    pos_enc_2 = torch.gather(pos_enc, 0, rel_pos_id)
                else:
                    pos_enc_2 = None

                #### Second type: Attn(pooled-q, pooled-kv)
                # construct rel_pos_id
                pos_id = pooled_pos_id
                rel_pos_id = self.construct_rel_pos_seq(
                    q_pos=pos_id, q_stride=q_stride,
                    k_pos=pos_id, k_stride=k_stride)

                # We scale Q at the end of the block
                q_stride *= scale_factor

                # gather relative positional encoding
                rel_pos_id = rel_pos_id[:, None] + zero_offset
                rel_pos_id = rel_pos_id.expand(rel_pos_id.size(0), d_model)
                pos_enc_1 = torch.gather(pos_enc, 0, rel_pos_id)

                pos_enc_list.append([pos_enc_1, pos_enc_2])
            return pos_enc_list
        else:
            raise NotImplementedError

    def seg_id_to_mat(self, seg_q, seg_k):
        """Convert `seg_id` to `seg_mat`."""
        seg_mat = torch.eq(torch.unsqueeze(seg_q, -1), torch.unsqueeze(seg_k, -2))

        # Treat [cls] as in the same segment as both A & B
        cls_mat = torch.unsqueeze(torch.eq(seg_q, self.net_config.seg_id_cls), -1) | \
                  torch.unsqueeze(torch.eq(seg_k, self.net_config.seg_id_cls), -2)
        seg_mat = cls_mat | seg_mat

        return seg_mat

    def get_attn_mask(self, input_mask):
        return None if input_mask is None else input_mask[:, None, None, :]

    def get_fresh_attn_tuple(self, seg_id=None, input_mask=None):
        self.delta = 1  # Reset our little counter
        seg_mat = None if seg_id is None else self.seg_id_to_mat(seg_id, seg_id)

        attn_mask = self.get_attn_mask(input_mask)
        return self.pos_enc, seg_mat, input_mask, attn_mask, self.func_mask

    def stride_pool(self, tensor, axis):
        """Perform pooling by stride slicing the tensor along the given axis."""
        if tensor is None:
            return None

        net_config = self.net_config
        if isinstance(tensor, (tuple, list)):
            ndims = tensor[0].dim()
        else:
            ndims = tensor.dim()
        axis = axis % ndims

        enc_slice = []
        for i in range(ndims):
            if i == axis:
                if net_config.separate_cls and net_config.truncate_seq:
                    enc_slice.append(slice(None, -1, 2))
                else:
                    enc_slice.append(slice(None, None, 2))
                break
            else:
                enc_slice.append(slice(None))

        if net_config.separate_cls:
            cls_slice = []
            for i in range(ndims):
                if i == axis:
                    cls_slice.append(slice(None, 1))
                    break
                else:
                    cls_slice.append(slice(None))

        def _pool_func(enc):
            # separate_cls = True
            #   trunc = False
            #     [0 1 2 3 4 5 6 7] => [0] & [1 2 3 4 5 6 7] => [0] & [1 3 5 7]
            #     [0 1 3 5 7] => [0] & [1 3 5 7] => [0] & [1 5]
            #     [0 1 5] => [0] & [1 5] =>  [0] & [1]
            #   trunc = True
            #     [0 1 2 3 4 5 6 7] => [0] & [1 2 3 4 5 6] => [0] & [1 3 5]
            #     [0 1 3 5] => [0] & [1 3] => [0] & [1]
            #     [0 1] => [0] & [] => [0]
            # separate_cls = False
            #   [0 1 2 3 4 5 6 7] => [0 2 4 6]
            #   [0 2 4 6] => [0 4]
            #   [0 4] => [0]

            if net_config.separate_cls:
                enc = torch.cat([enc[cls_slice], enc], axis=axis)
            return enc[enc_slice]

        if isinstance(tensor, (tuple, list)):
            return list(map(_pool_func, tensor))
        else:
            return _pool_func(tensor)

    def pool_tensor(self, tensor, mode="mean", stride=(2, 1)):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        net_config = self.net_config
        ndims = tensor.dim()
        if net_config.separate_cls:
            if net_config.truncate_seq:
                tensor = torch.cat([tensor[:, :1], tensor[:, :-1]], dim=1)
            else:
                tensor = torch.cat([tensor[:, :1], tensor], dim=1)

        assert ndims == 2 or ndims == 3 or ndims == 4

        if ndims == 2:
            tensor = tensor[:, None, :, None]
        elif ndims == 3:
            tensor = tensor[:, None, :, :]

        if mode == "mean":
            tensor = F.avg_pool2d(
                tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = F.max_pool2d(
                tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -F.max_pool2d(
                -tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError
        if ndims == 2:
            tensor = tensor.squeeze(-1).squeeze(1)
        elif ndims == 3:
            tensor = tensor.squeeze(1)

        return tensor

    def pre_attn_pooling(self, output, attn_struct):
        pos_enc, seg_mat, input_mask, attn_mask, func_mask = attn_struct
        net_config = self.net_config
        ret_dict = {}
        if net_config.pool_q_only:
            if net_config.attention_type == "factorized":
                pos_enc = self.stride_pool(pos_enc[:2], 0) + pos_enc[2:]
            seg_mat = self.stride_pool(seg_mat, 1)
            func_mask = self.stride_pool(func_mask, 0)
            output = self.pool_tensor(output, mode=net_config.pooling_type)
        else:
            self.delta *= 2
            if net_config.attention_type == "factorized":
                pos_enc = self.stride_pool(pos_enc, 0)
            seg_mat = self.stride_pool(seg_mat, 1)
            seg_mat = self.stride_pool(seg_mat, 2)
            func_mask = self.stride_pool(func_mask, 1)
            func_mask = self.stride_pool(func_mask, 2)
            input_mask = self.pool_tensor(input_mask, mode="min")
            output = self.pool_tensor(output, mode=net_config.pooling_type)
        attn_mask = self.get_attn_mask(input_mask)
        attn_struct = (pos_enc, seg_mat, input_mask, attn_mask, func_mask)
        return output, attn_struct, ret_dict

    def post_attn_pooling(self, attn_struct):
        net_config = self.net_config
        pos_enc, seg_mat, input_mask, attn_mask, func_mask = attn_struct

        if net_config.pool_q_only:
            self.delta *= 2
            if net_config.attention_type == "factorized":
                pos_enc = pos_enc[:2] + self.stride_pool(pos_enc[2:], 0)
            seg_mat = self.stride_pool(seg_mat, 2)
            func_mask = self.stride_pool(func_mask, 1)
            input_mask = self.pool_tensor(input_mask, mode="min")
        attn_mask = self.get_attn_mask(input_mask)
        attn_struct = (pos_enc, seg_mat, input_mask, attn_mask, func_mask)

        return attn_struct
