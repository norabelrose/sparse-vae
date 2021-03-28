from .AutoencoderCallback import *


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

        samples = langmodel.sample(self.sample_max_len, self.num_samples, temperature=self.sampling_temperature)
        if samples is None:
            return

        logger = logger.experiment
        tokenizer = trainer.datamodule.tokenizer

        # List of Markov chain MLM iterations
        if isinstance(samples, list):
            samples = [tokenizer.decode_batch(sample.tolist(), skip_special_tokens=False) for sample in samples]

            for batch in zip(*samples):
                buffer = ""
                for i, iteration in enumerate(batch):
                    if i == 0:
                        buffer += "Original sample:\n"
                    else:
                        buffer += f"Iteration {i}:\n"

                    buffer += iteration + "\n\n"

                logger.add_text("unconditional_sample", buffer, global_step=langmodel.global_step)

        # Normal case
        else:
            samples = samples.tolist()  # Tensor -> List of lists of ints
            samples = tokenizer.decode_batch(samples, skip_special_tokens=False)  # List of strings

            for sample in samples:
                logger.add_text("unconditional_sample", sample, global_step=langmodel.global_step)
