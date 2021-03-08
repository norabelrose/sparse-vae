from dataclasses import dataclass  # noqa; here to silence PyCharm linter bug
from .Autoencoder import *
from .Quantizer import *


@dataclass
class QuantizedVAEHparams(AutoencoderHparams):
    codebook_size: int = 8192
    beta: float = 0.25
    use_kmeans_codebook_updates: bool = True


class QuantizedVAE(LanguageModel, ABC):
    @property
    def quantizing(self) -> bool:
        return True
        # return not self.training or self.trainer.current_epoch > 0

    # def validation_epoch_end(self, outputs: List[Any]) -> None:
    #     if self.trainer.running_sanity_check or not self.hparams.use_kmeans_codebook_updates:
    #         return

    #     self.update_codebook_kmeans()
