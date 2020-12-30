from argparse import ArgumentParser
from pytorch_lightning import Trainer
from .Autoencoder import Autoencoder

if __name__ == "__main__":
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)
    parser = Autoencoder.add_model_specific_args(parser)
    args = parser.parse_args()

    vae = Autoencoder(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(vae)
