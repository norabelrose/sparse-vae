from argparse import ArgumentParser, Namespace
from pytorch_lightning import Trainer
from typing import Mapping, MutableMapping, Sequence
from .Autoencoder import Autoencoder
from .funnel_transformers.FunnelForPreTraining import FunnelForPreTraining
from .training.Datasets import ProjectGutenbergDataModule
from .training.FunnelPreTrainingDataModule import FunnelPreTrainingDataModule


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
    hparams = type(defaults)()

    # Recursively search for a hyperparameter with the appropriate name
    def search_for_key(key: str, mapping: Mapping) -> bool:
        if key in mapping:
            hparams[key] = value
            return True
        else:
            for subdict in filter(lambda x: isinstance(x, MutableMapping), mapping.values()):
                if search_for_key(key, subdict):
                    return True

            return False

    for argname, value in vars(args).items():
        assert search_for_key(argname, defaults), f"Couldn't find hyperparameter matching '{argname}'"

    return hparams


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
        model = FunnelForPreTraining(get_hparam_dict_from_args(args, FunnelForPreTraining.default_hparams))
    elif args.command == 'train':
        print("Training a Text VAE...")
        data = ProjectGutenbergDataModule()
        model = Autoencoder(get_hparam_dict_from_args(args, Autoencoder.default_hparams))
    else:
        raise NotImplementedError

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data)
