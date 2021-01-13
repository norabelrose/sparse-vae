from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pytorch_lightning import Trainer, LightningDataModule
from typing import Mapping, MutableMapping, Sequence
from .text_vae import Autoencoder
from .text_vae import FunnelForPreTraining
from .text_vae import ProjectGutenbergDataModule
from .text_vae import FunnelPreTrainingDataModule


def add_args_from_hparam_defaults(argparser: ArgumentParser, defaults: Mapping):
    for param, default in defaults.items():
        # For, i.e. Autoencoder.block_sizes
        if isinstance(default, Sequence):
            argparser.add_argument("--" + param, nargs='+', type=type(default[0]))

        # Recursively add arguments from nested dictionaries, like Autoencoder.default_hparams.funnel_hparams
        elif isinstance(default, Mapping):
            add_args_from_hparam_defaults(argparser, default)

        elif (default_type := type(default)) in (bool, float, int, str):
            argparser.add_argument("--" + param, type=default_type, default=default)

    return argparser

def get_hparam_dict_from_args(args: Namespace, defaults: MutableMapping):
    hparams = deepcopy(defaults)

    for argname, value in vars(args).items():
        if value is None:
            continue

        # Recursively search for a hyperparameter with the appropriate name
        def search_for_key(key: str, mapping: Mapping) -> bool:
            if key in mapping:
                mapping[key] = value
                return True
            else:
                for subdict in filter(lambda x: isinstance(x, MutableMapping), mapping.values()):
                    if search_for_key(key, subdict):
                        return True
        
        search_for_key(argname, hparams)


    return hparams


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = LightningDataModule.add_argparse_args(parser)

    subparsers = parser.add_subparsers(dest='command')
    finetune = subparsers.add_parser('finetune-funnel')
    train = subparsers.add_parser('train')

    add_args_from_hparam_defaults(finetune, FunnelForPreTraining.default_hparams)
    add_args_from_hparam_defaults(train, Autoencoder.default_hparams)

    args = parser.parse_args()

    if args.command == 'finetune-funnel':
        print("Finetuning a pretrained Funnel Transformer for Performer attention...")
        data = FunnelPreTrainingDataModule.from_argparse_args(args)
        hparams = get_hparam_dict_from_args(args, FunnelForPreTraining.default_hparams)
        model = FunnelForPreTraining(hparams)
    elif args.command == 'train':
        print("Training a Text VAE...")
        data = ProjectGutenbergDataModule.from_argparse_args(args)
        model = Autoencoder(get_hparam_dict_from_args(args, Autoencoder.default_hparams))
    else:
        raise NotImplementedError

    trainer = Trainer.from_argparse_args(args, accumulate_grad_batches=500)
    trainer.fit(model, datamodule=data)
