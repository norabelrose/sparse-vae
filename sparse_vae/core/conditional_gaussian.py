from torch import nn, Tensor
from torch.distributions import Normal
from typing import *


class ConditionalGaussian(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_initialized: bool = False, bias: bool = True):
        super(ConditionalGaussian, self).__init__()

        linear = nn.Linear(in_features, out_features * 2, bias=bias)
        if zero_initialized:
            linear.weight.data.zero_()
            if bias:
                linear.bias.data.zero_()

        self.linear = linear

    def forward(self, x: Tensor, get_kl: bool = False) -> Union[Normal, Tuple[Normal, Tensor]]:
        mu, logvar = self.linear(x).chunk(2, dim=-1)
        var = logvar.exp()

        # We do NOT validate the parameters here because this raises an error if any of the sigma values
        # are exactly zero. This should yield an infinite KL divergence and therefore an infinite loss,
        # but the AMP grad scaler will take care of that.
        gaussian = Normal(loc=mu, scale=var.sqrt(), validate_args=False)
        if get_kl:
            kl = 0.5 * (mu ** 2 + var - logvar - 1.0)
            return gaussian, kl
        else:
            return gaussian
