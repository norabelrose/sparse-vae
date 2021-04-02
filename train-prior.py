from pytorch_lightning import seed_everything
from text_vae import *
import sys


def main(args):
    version_name = args[1]

    seed_everything(7295)  # Reproducibility
    config = OmegaConf.create({
        # Override Trainer defaults but still allow them to be overridden by the command line
        'trainer': {
            'accumulate_grad_batches': 4,
            'precision': 16,
            'num_sanity_val_steps': 2
        }
    })

    sampler = QuantizedVAESampler.for_vae(version_name)

    config.data = OmegaConf.structured(QuantizedLatentDataModuleHparams)
    config.prior = OmegaConf.structured(TransformerHparams)

    config.merge_with_dotlist(args[2:])
    sampler.train_priors(config)


if __name__ == "__main__":
    main(sys.argv)
