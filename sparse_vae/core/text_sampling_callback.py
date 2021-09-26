from dataclasses import asdict, dataclass
from pytorch_lightning import Callback
from torchtext.data.metrics import bleu_score
from typing import *


@dataclass
class TextSamplingCallback(Callback):
    num_reconstructions: int = 1
    sample_max_len: int = 512
    train_step_interval: int = 500

    def on_train_batch_end(self, trainer, langmodel, outputs, batch, batch_idx, dataloader_idx):
        cur_step = langmodel.global_step
        if cur_step % self.train_step_interval != 0 or not outputs:
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

        if 'posterior' in outputs:
            z = outputs['posterior'].mean[0, None]
        elif 'z' in outputs:
            z = outputs['z'][0].unsqueeze(0)
        else:
            z = None

        if z is not None:
            langmodel.eval()
            reconstructions = langmodel.sample(self.sample_max_len, z.shape[0], z=z, temperature=0.7)
            langmodel.train()

            original = batch['token_ids'][0]
            original_len = min(len(original), self.sample_max_len)
            original_str = tokenizer.decode(original[:original_len].tolist())
            reconstructed_strs = tokenizer.decode_batch(reconstructions.tolist())
            langmodel.log('train_bleu', bleu_score(
                [x.split(' ') for x in reconstructed_strs], [[original_str.split(' ')]],
                max_n=2, weights=[0.5] * 2
            ), on_step=True)

            logged_msg = "**Original**:  \n" + original_str
            for i, reconstruction in enumerate(reconstructed_strs, start=1):
                logged_msg += f"  \n**Reconstruction {i}**:  \n" + reconstruction

            logger.add_text('reconstruction', logged_msg, global_step=langmodel.global_step)

    def on_load_checkpoint(self, ckpt_state: Dict[str, Any]):
        self.__dict__.update(ckpt_state)

    def on_save_checkpoint(self, trainer, pl_module, ckpt_state) -> Dict[str, Any]:
        return asdict(self)
