from dataclasses import asdict, dataclass
from pytorch_lightning import Callback
from torch.distributions import Normal
from typing import *


@dataclass
class TextSamplingCallback(Callback):
    num_reconstructions: int = 2
    sample_max_len: int = 512
    train_step_interval: int = 1000

    def on_train_batch_end(self, trainer, langmodel, outputs, batch, batch_idx, dataloader_idx):
        cur_step = langmodel.global_step
        if cur_step % self.train_step_interval != 0:
            return

        logger = trainer.logger
        if not logger:
            return

        langmodel.eval()
        unconditional = langmodel.sample(self.sample_max_len, 1)
        langmodel.train()
        if unconditional is None:
            return

        logger = logger.experiment
        tokenizer = trainer.datamodule.tokenizer
        unconditional = tokenizer.decode_batch(unconditional.tolist())  # List of strings

        for sample in unconditional:
            logger.add_text("unconditional_sample", sample, global_step=langmodel.global_step)

        # Weirdly PL wraps the actual training_step output in two lists and a dict
        outputs = outputs[0][0]
        if 'extra' in outputs:
            outputs = outputs['extra']

        if 'posterior' in outputs:
            posterior = outputs['posterior']
            posterior = Normal(loc=posterior.loc[0], scale=posterior.scale[0])
            langmodel.eval()

            # This adds a new leading dimension for the different samples from the posterior, but we now
            # squeeze away the original batch dimension, so to the decoder the different latent samples
            # just look like different elements in a batch
            z = posterior.rsample([self.num_reconstructions, 1])
            reconstructions = langmodel.sample(self.sample_max_len, z.shape[0], z=z)
            langmodel.train()

            original = batch['token_ids'][0]
            original_len = min(len(original), self.sample_max_len)
            logged_msg = "**Original**:  \n" + tokenizer.decode(original[:original_len].tolist())
            for i, reconstruction in enumerate(reconstructions, start=1):
                logged_msg += f"  \n**Reconstruction {i}**:  \n" + tokenizer.decode(reconstruction.tolist())

            logger.add_text('reconstruction', logged_msg, global_step=langmodel.global_step)

    def on_load_checkpoint(self, ckpt_state: Dict[str, Any]):
        self.__dict__.update(ckpt_state)

    def on_save_checkpoint(self, trainer, pl_module, ckpt_state) -> Dict[str, Any]:
        return asdict(self)
