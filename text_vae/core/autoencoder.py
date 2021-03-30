from abc import abstractmethod
from torch.distributions import Normal
from .language_model import *
from ..train_callbacks import KLAnnealing, ReconstructionSampler


@dataclass
class AutoencoderHparams(LanguageModelHparams, ABC):
    latent_depth: int = 16  # Depth of the latent tensors/vectors


# Abstract base classes for autoencoders with continuous latent spaces
@dataclass
class ContinuousVAEHparams(AutoencoderHparams, ABC):
    kl_weight: float = 1.0

class ContinuousVAE(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super(LanguageModel, self).__init__(hparams)

        # Create the standard diagonal Gaussian prior for the first layer
        self.register_buffer('prior_mu', torch.zeros(hparams.latent_depth))
        self.register_buffer('prior_sigma', torch.ones(hparams.latent_depth))

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return callbacks + [EarlyStopping(monitor='val_loss', mode='min'), KLAnnealing(), ReconstructionSampler()]

    # Workaround for the fact that Distribution objects don't have a .to() method
    def get_base_prior(self) -> Normal:
        return Normal(self.prior_mu, self.prior_sigma)

    # Called by AggressiveEncoderTraining callback
    @abstractmethod
    def decoder_requires_grad_(self, requires_grad: bool):
        raise NotImplementedError
