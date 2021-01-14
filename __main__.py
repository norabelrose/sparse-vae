from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from .text_vae import Autoencoder, AutoencoderHparams
from .text_vae import FunnelForPreTraining, FunnelTransformerHparams
from .text_vae import ProjectGutenbergDataModule
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

        config.funnel = OmegaConf.structured(FunnelTransformerHparams)
        config.merge_with_dotlist(args[2:])  # Skip both the application name and the subcommand

        data = FunnelPreTrainingDataModule()
        model = FunnelForPreTraining(config.funnel)

    elif command == 'train':
        print("Training a Text VAE...")

        config.vae = OmegaConf.structured(AutoencoderHparams)
        config.merge_with_dotlist(args[2:])

        data = ProjectGutenbergDataModule()
        model = Autoencoder(config.vae)
    else:
        raise NotImplementedError

    trainer = Trainer(**config.trainer)
    trainer.fit(model, datamodule=data)
