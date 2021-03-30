from torch import nn, Tensor
from torch.distributions import Normal


class ConditionalGaussian(nn.Module):
    def __init__(self, in_features: int, out_features: int, reduce_dim: int = None, zero_initialized: bool = False):
        super(ConditionalGaussian, self).__init__()

        linear = nn.Linear(in_features, out_features * 2)
        if zero_initialized:
            linear.bias.data.zero_()
            linear.weight.data.zero_()

        self.linear = nn.Sequential(nn.GELU(), linear)
        self.reduce_dim = reduce_dim

    def forward(self, x: Tensor, temperature: float = 1.0) -> Normal:
        mu, logsigma = self.linear(x).chunk(2, dim=-1)
        if self.reduce_dim is not None:
            mu = mu.mean(dim=self.reduce_dim)
            logsigma = logsigma.mean(dim=self.reduce_dim)

        return Normal(loc=mu, scale=logsigma.exp() * temperature)
