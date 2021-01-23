from pytorch_lightning.metrics import Metric
from torch import Tensor
import math
import torch


class MutualInformation(Metric):
    def __init__(self):
        super().__init__()

        self.add_state('running_total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('denominator', default=torch.tensor(0), dist_reduce_fx='sum')

    def compute(self):
        return self.running_total / self.denominator

    def update(self, posterior: torch.distributions.Normal, z: Tensor):  # noqa
        mu, logvar = posterior.mean, posterior.variance.log()
        x_batch, nz = mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = posterior.variance

        # (z_batch, x_batch, nz)
        dev = z - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - 0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_density.logsumexp(dim=1) - math.log(log_density.shape[1])

        self.running_total += (neg_entropy - log_qz.mean(-1))
        self.denominator += 1
