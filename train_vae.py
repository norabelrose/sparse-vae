from sparse_vae import TextDataModule, TransformerVAE
from pytorch_lightning.utilities.cli import LightningCLI    # noqa

LightningCLI(
    TransformerVAE, datamodule_class=TextDataModule, seed_everything_default=7295
)
