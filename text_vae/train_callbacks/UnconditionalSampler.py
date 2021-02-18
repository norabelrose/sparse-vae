from text_vae.core.LanguageModel import LanguageModel
from .AutoencoderCallback import *


@dataclass
class UnconditionalSampler(AutoencoderCallback):
    num_samples: int = 1
    sample_max_len: int = 192
    sampling_temperature: float = 0.85  # Taken from Very Deep VAEs paper
    train_step_interval: int = 1000

    def on_train_batch_end(self, trainer, langmodel: LanguageModel, outputs, batch, batch_idx, dataloader_idx):
        cur_step = langmodel.global_step
        if cur_step % self.train_step_interval != 0 or not langmodel.should_unconditionally_sample():
            return

        samples = langmodel.sample(self.sample_max_len, self.num_samples, temperature=self.sampling_temperature)
        samples = samples.tolist()  # Tensor -> List of lists of ints
        samples = langmodel.tokenizer.decode_batch(samples, skip_special_tokens=True)  # List of strings

        logger = trainer.logger.experiment
        for sample in samples:
            logger.add_text("unconditional_sample", sample, global_step=langmodel.global_step)
