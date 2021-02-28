from dataclasses import dataclass  # noqa; here to silence PyCharm linter bug
from .Autoencoder import *
from .Quantizer import *


@dataclass
class QuantizedVAEHparams(AutoencoderHparams):
    codebook_size: int = 512
    beta: float = 0.25
    use_kmeans_codebook_updates: bool = True


class QuantizedVAE(LanguageModel, ABC):
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        if self.trainer.running_sanity_check or not self.hparams.use_kmeans_codebook_updates:
            return

        self.update_codebook_kmeans()

    @torch.no_grad()
    def update_codebook_kmeans(self):
        self.print("\nPerforming K means codebook update...")

        # Do encoder forward passes through the entire training dataset in order to gather the soft codes
        loader = self.trainer.train_dataloader
        loader = tqdm(islice(loader, len(loader)), desc='Gathering encoder outputs', total=len(loader))
        observed_codes = [self.forward(
            {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()},
            quantize=False
        ).soft_codes for batch in loader]

        self.quantizer.perform_kmeans_update(torch.cat(observed_codes, dim=0))
        self.code_frequencies.zero_()
