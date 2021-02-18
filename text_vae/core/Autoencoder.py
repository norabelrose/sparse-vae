from abc import abstractmethod
from torch.distributions import Normal
from ..AutoencoderDataModule import *
from .LanguageModel import *


@dataclass
class AutoencoderHparams(LanguageModelHparams, ABC):
    latent_depth: int = 16  # Depth of the latent tensors/vectors

class Autoencoder(LanguageModel, ABC):
    @abstractmethod
    def compute_posteriors(self, batch: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def extract_posteriors_for_dataset(self, datamodule: AutoencoderDataModule):
        batch_sz = datamodule.batch_size
        dataset = datamodule.dataset
        dataset.set_format('torch')  # Makes the dataset yield PyTorch tensors

        def get_posteriors(batch: Dict[str, list]) -> Dict[str, list]:
            batch = [dict(zip(batch, x)) for x in zip(*batch.values())]  # dict of lists -> list of dicts
            batch = datamodule.collate(batch)
            return {'posteriors': self.compute_posteriors(batch)}

        print(f"Extracting posteriors over the latent space for dataset '{datamodule.hparams.dataset_name}'...")
        datamodule.dataset = dataset.map(get_posteriors, batched=True, batch_size=batch_sz, load_from_cache_file=False)


# Abstract base classes for autoencoders with continuous latent spaces
@dataclass
class ContinuousAutoencoderHparams(AutoencoderHparams, ABC):
    kl_weight: float = 1.0

class ContinuousAutoencoder(Autoencoder):
    def __init__(self, hparams: DictConfig):
        super(Autoencoder, self).__init__(hparams)

        # Create the standard diagonal Gaussian prior for the first layer
        self.register_buffer('prior_mu', torch.zeros(hparams.latent_depth))
        self.register_buffer('prior_sigma', torch.ones(hparams.latent_depth))

    # Workaround for the fact that Distribution objects don't have a .to() method
    def get_base_prior(self) -> Normal:
        return Normal(self.prior_mu, self.prior_sigma)

    # Called by AggressiveEncoderTraining callback
    @abstractmethod
    def decoder_requires_grad_(self, requires_grad: bool):
        raise NotImplementedError
