from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from .text_vae import Autoencoder, AutoencoderHparams
from .text_vae import FunnelForPreTraining, FunnelTransformerHparams
from .text_vae import TextVaeDataModule, TextVaeDataModuleHparams
from .text_vae import FunnelPreTrainingDataModule
import sys
import torch


if __name__ == "__main__":
    args = sys.argv
    command = args[1]

    config = OmegaConf.create({
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            'gpus': int(torch.cuda.is_available()),
            'precision': 16
        }
    })
    if command == 'finetune-funnel':
        print("Finetuning a pretrained Funnel Transformer for Performer attention...")

        config.data = OmegaConf.structured(TextVaeDataModuleHparams)
        config.funnel = OmegaConf.structured(FunnelTransformerHparams)
        config.merge_with_dotlist(args[2:])  # Skip both the application name and the subcommand

        data = FunnelPreTrainingDataModule(hparams=config.data)
        model = FunnelForPreTraining(config.funnel)

    elif command == 'train':
        print("Training a Text VAE...")

        config.data = OmegaConf.structured(TextVaeDataModuleHparams)
        config.vae = OmegaConf.structured(AutoencoderHparams)
        config.merge_with_dotlist(args[2:])

        data = TextVaeDataModule(hparams=config.data)
        model = Autoencoder(config.vae)
    else:
        raise NotImplementedError

    trainer = Trainer(**config.trainer)
    trainer.fit(model, datamodule=data)
