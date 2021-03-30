from .autoencoder_callback import *
from torch.distributions import Categorical


@dataclass
class ReconstructionSampler(AutoencoderCallback):
    num_samples: int = 1
    train_step_interval: int = 1000

    def on_train_batch_end(self, trainer, autoencoder, outputs, batch, batch_idx, _):
        if autoencoder.global_step % self.train_step_interval != 0:
            return

        logger = trainer.logger
        if not logger:
            return

        # Weirdly PL wraps the actual training_step output in two lists and a dict
        outputs = outputs[0][0]
        if 'extra' in outputs:
            outputs = outputs['extra']

        if (logits := outputs.get('logits')) is not None:
            dist = Categorical(logits=logits)
        elif outputs := outputs.get('output'):
            dist = outputs.p_of_x_given_z
        else:
            return

        tokenizer = trainer.datamodule.tokenizer

        # We're doing a masked language modeling / denoising autoencoder task
        if (originals := batch.get('labels')) is not None:
            originals = originals[:self.num_samples].tolist()
            originals = tokenizer.decode_batch(originals)

        model_inputs = batch['token_ids'][:self.num_samples].tolist()  # Tensor -> List of lists of ints
        model_inputs = tokenizer.decode_batch(model_inputs, skip_special_tokens=False)  # List of strings
        if not originals:
            originals = model_inputs

        reconstruction_ids = dist.logits[:self.num_samples].argmax(dim=-1).tolist()
        reconstructions = tokenizer.decode_batch(reconstruction_ids, skip_special_tokens=False)

        logger = logger.experiment
        for original, model_input, reconstruction in zip(originals, model_inputs, reconstructions):
            logged_msg = "Original:  \n" + original
            if model_input != original:
                logged_msg += "  \nMasked:  \n" + model_input
            logged_msg += "  \nReconstruction:  \n" + reconstruction

            logger.add_text('reconstruction', logged_msg, global_step=autoencoder.global_step)
