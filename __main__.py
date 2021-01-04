from argparse import ArgumentParser
from pytorch_lightning import Trainer
from typing import Sequence
from .Autoencoder import Autoencoder
from .funnel_transformers.FunnelForPreTraining import FunnelForPreTraining
from .training.Datasets import ProjectGutenbergDataModule
from .training.FunnelPreTrainingDataModule import FunnelPreTrainingDataModule


def add_args_from_hparam_defaults(argparser: ArgumentParser, defaults: dict):
    for param, default in defaults:
        # For, i.e. Autoencoder.block_sizes
        if isinstance(default, Sequence):
            argparser.add_argument("--" + param, nargs='+', type=type(default[0]))

        elif (default_type := type(default)) in (bool, float, int, str):
            argparser.add_argument("--" + param, type=default_type, default=default)

    return argparser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    subparsers = parser.add_subparsers(dest='command')
    finetune = subparsers.add_parser('finetune-funnel')
    train = subparsers.add_parser('train')

    add_args_from_hparam_defaults(finetune, FunnelForPreTraining.default_hparams)
    add_args_from_hparam_defaults(train, Autoencoder.default_hparams)

    args = parser.parse_args()

    if args.command == 'finetune-funnel':
        print("Finetuning a pretrained Funnel Transformer for Performer attention...")
        data = FunnelPreTrainingDataModule(batch_size=48)
        model = FunnelForPreTraining(vars(args))
    elif args.command == 'train':
        print("Training a Text VAE...")
        data = ProjectGutenbergDataModule()
        model = Autoencoder(vars(args))
    else:
        raise NotImplementedError

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data)
