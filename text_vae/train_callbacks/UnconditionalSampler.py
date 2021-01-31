from ..Autoencoder import Autoencoder
from .AutoencoderCallback import *


@dataclass
class UnconditionalSampler(AutoencoderCallback):
    num_samples: int = 5
    sample_max_len: int = 192
    sampling_temperature: float = 0.85  # Taken from Very Deep VAEs paper

    def on_train_epoch_end(self, trainer, autoencoder: Autoencoder, outputs):
        samples = autoencoder.sample(self.sample_max_len, self.num_samples, temperature=self.sampling_temperature)
        samples = samples.tolist()  # Tensor -> List of lists of ints
        samples = autoencoder.tokenizer.decode_batch(samples, skip_special_tokens=True)  # List of strings

        logger = trainer.logger.experiment
        for sample in samples:
            logger.add_text("sample_epoch", sample, global_step=trainer.current_epoch)
