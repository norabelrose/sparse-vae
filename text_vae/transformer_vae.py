from .core import (
    ConditionalGaussian, ContinuousVAE, ContinuousVAEHparams, PaddedTensor, Perceiver,
    Transformer, TransformerHparams
)
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from omegaconf import DictConfig
from torch import nn
from typing import *
import torch


@dataclass
class TransformerVAEHparams(TransformerHparams, ContinuousVAEHparams):
    latent_depth: int = 32
    num_latent_vectors: int = 16


class TransformerVAE(Transformer, ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        self.example_input_array = None

        self.encoder_input_layer = deepcopy(self.input_layer)
        self.encoder_input_layer[0].weight = self.input_layer[0].weight

        self.encoder = Perceiver(
            num_layers=4, num_latents=32, d_model=hparams.d_model, bottleneck_width=hparams.num_latent_vectors
        )
        self.q_of_z_given_x = ConditionalGaussian(hparams.d_model, hparams.latent_depth)
        self.z_to_hidden = nn.Linear(hparams.latent_depth, hparams.d_model)

    def decoder_params(self) -> Iterable[nn.Parameter]:
        return chain(
            self.context_layer.parameters(),
            self.decoder_layers.parameters(),
            self.output_layer.parameters()
        )

    def training_step(self, batch: Dict[str, PaddedTensor], batch_index: int, stage: str = 'train'):
        original = batch['token_ids']
        x = self.input_layer(original)
        encoder_out = self.encoder(x)

        z, kl = self.sample_z(encoder_out, token_counts=batch['token_count'], stage=stage)
        z_hidden = self.z_to_hidden(z)

        use_cross_attn = self.hparams.num_latent_vectors > 1
        for layer in self.decoder_layers:
            if use_cross_attn:
                x = layer(x, context=z_hidden)
            else:
                x = torch.cat([x[..., 0, None, :] + z_hidden, x[..., 1:, :]], dim=-2)
                x = layer(x)

        logits = self.output_layer(x)
        nll = self.get_nll(logits[..., :-1, :], original[..., 1:], token_counts=batch['token_count'],
                           word_counts=batch['word_count'])
        loss = (nll + self.hparams.kl_weight * kl)
        if stage == 'train':
            return {'logits': logits, 'loss': loss} if loss.isfinite() else None
        elif stage == 'val':
            self.log('val_loss', nll + kl)

    def validation_step(self, batch: Dict[str, PaddedTensor], batch_index: int):
        return self.training_step(batch, batch_index, stage='val')

    def p_of_x_given_z(self, z, labels):
        for layer in self.decoder_layers:
            z = layer(z)

        logits = self.output_layer(z)
        return self.get_nll(logits, labels[..., 1:])

    @torch.cuda.amp.autocast()
    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        # Unconditional samples will be mostly garbage when we haven't annealed to the full KL weight
        if self.hparams.kl_weight < 1.0:
            return None

        z = torch.randn(batch_size, self.hparams.num_latent_vectors, self.hparams.latent_depth, device=self.device)
        z = self.z_to_hidden(z)
        z = PaddedTensor.unpadded(z)
        return super().sample(max_length, batch_size, initial_embedding=z, **kwargs)
