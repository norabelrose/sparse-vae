import torch
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange
from torch import nn, Tensor


@dataclass
class NystromRecurrentState:
    k_landmarks: Tensor
    q_landmarks: Tensor
    z_star: Tensor  # Pseudo-inverse of the q-k landmark interactions

    kernel1_denom: Tensor
    kernel2_denom: Tensor


class NystromAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_landmarks: int = 64, conv_kernel_size: int = None):
        super().__init__()

        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.qkv_linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

        # For fast recurrent autoregressive decoding
        self.cached_recurrent_state = None

        self.conv = nn.Conv2d(
            in_channels = num_heads,
            out_channels = num_heads,
            kernel_size = (conv_kernel_size, 1),
            padding = (conv_kernel_size // 2, 0),
            bias = False,
            groups = num_heads
        ) if conv_kernel_size else None

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        q, k, v = (linear(x) for x, linear in zip((q, k, v), self.qkv_linears))
        q, k, v = (rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads) for x in (q, k, v))
        (_, head_dim, n, d_model), m = q.shape, self.num_landmarks

        q = q / head_dim ** 0.25
        k = k / head_dim ** 0.25

        # Just use full softmax attention if the sequence length is less than or equal to the number of landmarks
        if n <= m:
            scores = q @ k.transpose(-1, -2)
            scores -= 1e9 * ~mask[:, None, None, :]
            output = F.softmax(scores, dim=-1) @ v
        else:
            # Pad so that sequence can be evenly divided into m landmarks
            remainder = n % m
            if remainder > 0:
                padding = m - (n % m)
                q, k, v = (F.pad(x, (0, 0, 0, padding), value=0) for x in (q, k, v))

                if mask is not None:
                    mask = F.pad(mask, (0, padding), value=False)

            if mask is not None:
                q, k = (x * mask[:, None, :, None] for x in (q, k))

            # Landmarks are average pooled segments of the sequence
            q_landmarks = q.unflatten(-2, (m, n // m)).mean(dim=-2)  # noqa
            k_landmarks = k.unflatten(-2, (m, n // m)).mean(dim=-2)  # noqa

            output = self._compute_attn_nystrom(q, k, v, q_landmarks, k_landmarks, mask)

        if self.conv:
            output += self.conv(v * mask[:, None, :, None] if mask is not None else v)

        output = rearrange(output, 'b h l d -> b l (h d)')
        return self.output_linear(output)

    # The core of the Nyström attention algorithm
    @staticmethod
    def _compute_attn_nystrom(q: Tensor, k: Tensor, v: Tensor, q_tilde: Tensor, k_tilde: Tensor, mask: Tensor = None):
        kernel_1 = q @ k_tilde.transpose(-1, -2)
        kernel_2 = q_tilde @ k_tilde.transpose(-1, -2)
        kernel_3 = q_tilde @ k.transpose(-1, -2)
        if mask is not None:
            kernel_3 = kernel_3 - 1e9 * (1 - mask[:, None, None, :])

        kernel_1, kernel_2, kernel_3 = (F.softmax(x, dim=-1) for x in (kernel_1, kernel_2, kernel_3))
        return kernel_1 @ pseudo_inv(kernel_2) @ (kernel_3 @ v)

# Approximate matrix pseudo-inverse that doesn't blow up if a matrix is singular. See Nyströmformer, pages 4-5
def pseudo_inv(a_s: Tensor, n_iter: int = 6):
    I = torch.eye(a_s.size(-1), device = a_s.device)  # noqa
    z = 1 / (a_s.abs().sum(dim = -2).max() * a_s.abs().sum(dim = -1).max()) * a_s.transpose(-1, -2)
    for _ in range(n_iter):
        a_sz = a_s @ z
        z = 0.25 * z @ (13 * I - a_sz @ (15 * I - a_sz @ (7 * I - a_sz)))

    return z
