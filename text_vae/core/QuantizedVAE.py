from dataclasses import dataclass  # noqa; here to silence PyCharm linter bug
from .Autoencoder import *
from .Quantizer import *


@dataclass
class QuantizedVAEHparams(AutoencoderHparams):
    codebook_size: int = 8192
    latent_depth: int = 64
    beta: float = 0.25
    quantize_in_first_epoch: bool = True
    use_kmeans_codebook_updates: bool = True


class QuantizedVAE(LanguageModel, ABC):
    @property
    def quantizing(self) -> bool:
        return self.hparams.quantize_in_first_epoch or not self.training or self.trainer.current_epoch > 0
