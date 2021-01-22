from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from benchmarks.LSTMAutoencoder import LSTMAutoencoder, LSTMAutoencoderHparams
from text_vae import Autoencoder, AutoencoderHparams
from text_vae import FunnelForPreTraining, FunnelTransformerHparams
from text_vae import TextVaeDataModule, TextVaeDataModuleHparams
from text_vae import FunnelPreTrainingDataModule
import sys
import torch


def main(args):
    command = args[1]

    # torch.autograd.set_detect_anomaly(True)
    gpu_available = torch.cuda.is_available()
    config = OmegaConf.create({
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            'auto_select_gpus': gpu_available,
            'gpus': int(gpu_available),
            'precision': 32
        }
    })
    data_class = TextVaeDataModule

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

    config.data = OmegaConf.structured(TextVaeDataModuleHparams)
    config.model = OmegaConf.structured(hparam_class)
    config.merge_with_dotlist(args[2:])

    data = data_class(hparams=config.data)
    model = model_class(config.model)

    trainer = Trainer(**config.trainer)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
