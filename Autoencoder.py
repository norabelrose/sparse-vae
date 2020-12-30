from argparse import ArgumentParser
from torch import Tensor
from typing import *
from .Decoder import Decoder
from .funnel_transformers.FunnelTransformer import FunnelTransformer
import pytorch_lightning as pl
import torch


class Autoencoder(pl.LightningModule):
    default_hparams = dict(
        d_model=768,                            # Embedding dimension
        latent_depth=16,                        # Depth of the latent tensors (dimensionality per token)
        attention_head_depth=64,                # The dimensionality of each attention head
        use_pretrained_encoder=True,
        copy_encoder_weights_to_decoder=True,
        condition_on_title=True,                # Whether to condition z0 on the document title
        use_autoregressive_decoding=False,
        use_performer_attention=False,          # See "Rethinking Attention with Performers" paper
        max_sequence_length=512,
        attention_dropout=0.1
    )

    # For the command line interface
    @classmethod
    def add_model_specific_args(cls, parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for param, default in cls.default_hparams:
            parser.add_argument("--" + param, type=type(default), default=default)

        parser.add_argument("--block_sizes", nargs='+', type=int)
        parser.add_argument("--scaling_factors", nargs='+', type=int)
        return parser

    def __init__(self, block_sizes: Sequence[int], scaling_factors: Sequence[int], **kwargs):
        super().__init__()

        # Now we can call `self.hparams`. save_hyperparameters() ensures the parameters are saved during training.
        kwargs = {**self.default_hparams, **kwargs}
        self.save_hyperparameters()

        funnel_hparams = dict(
            attention_type="factorized" if self.use_performer_attention else "rel_shift",
            block_sizes=self.block_sizes[0:3],
            max_position_embeddings=self.max_sequence_length,
            load_pretrained_weights=self.use_pretrained_encoder,
            return_block_outputs=True,
            use_performer_attention=self.use_performer_attention
        )

        # If copy_encoder_weights_to_decoder == True, we should load the weights once here and then hand it
        # off to both the encoder and the decoder
        decoder_funnel = None
        if self.hparams.use_pretrained_encoder:
            # self.encoder_funnel = PretrainedModelManager.get_model(self.hparams.block_sizes)
            if self.hparams.copy_encoder_weights_to_decoder:
                blocks_to_reset = range(3, len(self.hparams.block_sizes) - 3)
                decoder_funnel = self.encoder_funnel.inverted_copy(reinitialize_blocks=blocks_to_reset)
        else:
            self.encoder_funnel = FunnelTransformer(**funnel_hparams)
            if self.hparams.use_pretrained_encoder:
                self.encoder_funnel.load_pretrained_weights()

        self.decoder = Decoder(self.hparams, funnel_to_use=decoder_funnel)

    def forward(self, *args, **kwargs):
        pass

    # Returns the loss
    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> Tensor:
        states: List[Tensor] = self.encoder_funnel(batch)

    def configure_optimizers(self):
        # TODO: Add LR decay
        return torch.optim.AdamW(params=self.parameters(), lr=1e-4, weight_decay=0.01)

