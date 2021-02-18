from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import *

from torch import nn, Tensor
from torch.distributions import Normal, Gamma


class ConditionalDistribution(nn.Module, ABC):
    num_params = 2
    torch_distribution = None

    def __init__(self, in_features: int, out_features: int, reduce_dim: int = None, zero_initialized: bool = False):
        super(ConditionalDistribution, self).__init__()

        linear = nn.Linear(in_features, out_features * self.num_params)
        if zero_initialized:
            linear.bias.data.zero_()
            linear.weight.data.zero_()

        self.linear = nn.Sequential(nn.GELU(), linear)
        self.reduce_dim = reduce_dim

    def forward(self, x: Tensor, temperature: float = 1.0) -> Normal:
        params = self.linear(x).chunk(2, dim=-1)
        if self.reduce_dim is not None:
            params = [param.mean(dim=self.reduce_dim) for param in params]

        params = self.transform_parameters(params, temperature)  # noqa
        return self.torch_distribution(*params)

    # Convert the raw output of the Linear layer into something that we can pass to the PyTorch Distribution object
    @abstractmethod
    def transform_parameters(self, params: List[Tensor], temperature: float):
        raise NotImplementedError


class ConditionalGaussian(ConditionalDistribution):
    torch_distribution = Normal

    def transform_parameters(self, params: List[Tensor], temperature: float):
        return params[0], params[1].exp() * temperature  # mu, logsigma -> mu, sigma


class ConditionalGamma(ConditionalDistribution):
    torch_distribution = Gamma

    def transform_parameters(self, params: List[Tensor], temperature: float):
        return params[0].exp(), params[1].exp() * temperature  # logalpha, logbeta -> alpha, beta


@dataclass
class NormalGamma:
    mean_distribution: Normal
    precision_distribution: Gamma

    def log_prob(self, dist: Normal) -> Tensor:
        mu_log_prob = self.mean_distribution.log_prob(dist.mean)
        lambda_log_prob = self.precision_distribution.log_prob(dist.scale ** -2.0)
        return mu_log_prob + lambda_log_prob

    def rsample(self, sample_shape: Sequence[int] = None) -> Normal:
        mean = self.mean_distribution.rsample(sample_shape)
        precision = self.precision_distribution.rsample(sample_shape)

        return Normal(loc=mean, scale=precision ** -0.5)

class ConditionalNormalGamma:
    def __init__(self, in_features: int, out_features: int, reduce_dim: int = None, zero_initialized: bool = False):
        self.mu_distribution = ConditionalGaussian(in_features, out_features, reduce_dim, zero_initialized)
        self.sigma_distribution = ConditionalGamma(in_features, out_features, reduce_dim, zero_initialized)

    def forward(self, x: Tensor, temperature: float = 1.0) -> NormalGamma:
        return NormalGamma(
            self.mu_distribution(x, temperature),
            self.sigma_distribution(x, temperature)
        )
