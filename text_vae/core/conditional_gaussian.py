from torch import nn, Tensor
from torch.distributions import Normal
from typing import *


class ConditionalGaussian(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_initialized: bool = False):
        super(ConditionalGaussian, self).__init__()

        linear = nn.Linear(in_features, out_features * 2)
        if zero_initialized:
            linear.bias.data.zero_()
            linear.weight.data.zero_()

        self.linear = nn.Sequential(nn.GELU(), linear)

    def forward(self, x: Tensor, temperature: float = 1.0, get_kl: bool = False) -> Union[Normal, Tuple[Normal, Tensor]]:
        mu, logsigma = self.linear(x).chunk(2, dim=-1)
        sigma = logsigma.exp()

        gaussian = Normal(loc=mu, scale=sigma * temperature)
        if get_kl:
            # Analytical formula for the KL divergence p -> q, where p is a standard unit variance Gaussian
            kl = -0.5 + logsigma + 0.5 * (1.0 + mu ** 2) / (sigma ** 2)
            return gaussian, kl
        else:
            return gaussian
