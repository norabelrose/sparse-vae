import torch
from abc import abstractmethod
from torch.distributions import Normal
from .LanguageModel import *


@dataclass
class AutoencoderHparams(LanguageModelHparams, ABC):
    latent_depth: int = 16  # Depth of the latent tensors/vectors
    kl_weight: float = 1.0


class Autoencoder(LanguageModel, ABC):
    def __init__(self, hparams: DictConfig):
        super(Autoencoder, self).__init__(hparams)

        # Create the standard diagonal Gaussian prior for the first layer
        self.register_buffer('prior_mu', torch.zeros(hparams.latent_depth))
        self.register_buffer('prior_sigma', torch.ones(hparams.latent_depth))

    # Called by AggressiveEncoderTraining callback
    @abstractmethod
    def decoder_requires_grad_(self, requires_grad: bool):
        raise NotImplementedError

    # Called by UnconditionalSampler callback
    @abstractmethod
    def sample(self, max_length: int, count: int = 1, **kwargs):
        raise NotImplementedError

    # Workaround for the fact that Distribution objects don't have a .to() method
    def get_base_prior(self) -> Normal:
        return Normal(self.prior_mu, self.prior_sigma)
