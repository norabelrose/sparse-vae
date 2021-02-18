from functools import lru_cache
from survae.distributions import ConditionalDistribution
from survae.transforms.surjections import Surjection
from torch import Tensor
from typing import *
import torch


class Stride(Surjection):
    def __init__(self, distribution: ConditionalDistribution, stride: int = 2, dim: int = -2):
        super(Stride, self).__init__()
        assert dim in (-1, -2)

        self.distribution = distribution
        self.stride = stride
        self.dim = dim

    # Returns the strided tensor along with the log Jacobian determinant
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        strided = stride_at(x, stride=self.stride, dim=self.dim)
        leftover = stride_leftover(x, self.stride, self.dim)

        return strided, self.distribution.log_prob(leftover, context=strided)

    # Returns upsampled tensor
    def inverse(self, z) -> Tensor:
        new_shape = list(z.shape)
        new_shape[self.dim] *= self.stride

        output = torch.empty(*new_shape, device=z.device)
        stride_at(output, stride=self.stride, dim=self.dim).copy_(z)

        indices = leftover_indices(new_shape[self.dim], self.stride, z.device)
        output.index_copy_(self.dim, indices, self.distribution.sample(context=z))

        return output

    @property
    def stochastic_forward(self):
        return False

    @property
    def stochastic_inverse(self):
        return True

def stride_at(x: Tensor, stride: int, dim: int):
    slices = [slice(None, None) for _ in range(dim % x.ndim)] + [slice(None, None, stride)]
    return x[slices]

def stride_leftover(x: Tensor, stride: int, dim: int):
    grouped = x.unflatten(dim, (x.size(dim) // stride, stride))  # noqa
    return grouped.narrow(dim + 1, 1, stride - 1).flatten(dim, dim + 1)

@lru_cache(maxsize=None)
def leftover_indices(size: int, stride: int, device: torch.device):
    indices = torch.arange(size, device=device)
    return stride_leftover(indices, stride, 0)
