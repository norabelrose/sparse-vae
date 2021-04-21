from .autoencoder_callback import *


@dataclass
class UnconditionalSampler(AutoencoderCallback):
    num_samples: int = 1
    sample_max_len: int = 512
    sampling_temperature: float = 0.85  # Taken from Very Deep VAEs paper
    train_step_interval: int = 1000

    def on_train_batch_end(self, trainer, langmodel, outputs, batch, batch_idx, dataloader_idx):
        cur_step = langmodel.global_step
        if cur_step % self.train_step_interval != 0:
            return

        logger = trainer.logger
        if not logger:
            return

        langmodel.eval()
        samples = langmodel.sample(self.sample_max_len, self.num_samples, temperature=self.sampling_temperature)
        langmodel.train()
        if samples is None:
            return

        logger = logger.experiment
        tokenizer = trainer.datamodule.tokenizer

        samples = samples.tolist()  # Tensor -> List of lists of ints
        samples = [[token for token in sample if token != 0] for sample in samples]
        samples = tokenizer.decode_batch(samples, skip_special_tokens=False)  # List of strings

        for sample in samples:
            logger.add_text("unconditional_sample", sample, global_step=langmodel.global_step)
