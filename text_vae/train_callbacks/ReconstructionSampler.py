from .AutoencoderCallback import *
from ..Autoencoder import Autoencoder


@dataclass
class ReconstructionSampler(AutoencoderCallback):
    num_samples: int = 3

    def on_validation_batch_end(self, trainer, autoencoder: Autoencoder, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx > 0:
            return

        tokenizer = autoencoder.tokenizer
        original_ids = batch['token_ids'][:self.num_samples].tolist()  # Tensor -> List of lists of ints
        reconstruction_ids = outputs[0].logits[:self.num_samples].tolist()

        originals = tokenizer.decode_batch(original_ids)    # List of strings
        reconstructions = tokenizer.decode_batch(reconstruction_ids)

        logger = trainer.logger.experiment
        for original, reconstruction in zip(originals, reconstructions):
            logger.add_text('reconstruction', "Original:\n" + original + "\nReconstruction:\n" + reconstruction)
