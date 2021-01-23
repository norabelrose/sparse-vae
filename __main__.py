from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from benchmarks.LSTMAutoencoder import LSTMAutoencoder, LSTMAutoencoderHparams
from text_vae import AggressiveEncoderTraining
from text_vae import Autoencoder, AutoencoderHparams
from text_vae import FunnelForPreTraining, FunnelTransformerHparams
from text_vae import AutoencoderDataModule, AutoencoderDataModuleHparams
from text_vae import FunnelPreTrainingDataModule
import sys
import torch


def main(args):
    command = args[1]

    seed_everything(7295)  # Reproducibility

    # torch.autograd.set_detect_anomaly(True)
    gpu_available = torch.cuda.is_available()
    config = OmegaConf.create({
        'aggressive_encoder_training': False,
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            'auto_select_gpus': gpu_available,
            'gpus': int(gpu_available),
            'precision': 32
        }
    })
    data_class = AutoencoderDataModule

    if command == 'finetune-funnel':
        print("Finetuning a pretrained Funnel Transformer for Performer attention...")

        data_class = FunnelPreTrainingDataModule
        hparam_class = FunnelTransformerHparams
        model_class = FunnelForPreTraining

    elif command == 'train':
        print("Training a Text VAE...")

        hparam_class = AutoencoderHparams
        model_class = Autoencoder

    elif command == 'train-lstm':
        print("Training an LSTM VAE...")

        hparam_class = LSTMAutoencoderHparams
        model_class = LSTMAutoencoder

    else:
        raise NotImplementedError

    config.data = OmegaConf.structured(AutoencoderDataModuleHparams)
    config.model = OmegaConf.structured(hparam_class)
    config.merge_with_dotlist(args[2:])

    data = data_class(hparams=config.data)
    model = model_class(config.model)

    callbacks = None if not config.aggressive_encoder_training else [AggressiveEncoderTraining()]
    trainer = Trainer(**config.trainer, callbacks=callbacks)

    # Find the appropriate batch size for this machine and task
    if config.trainer.auto_scale_batch_size:
        trainer.tune(model, datamodule=data)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
