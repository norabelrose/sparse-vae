from .core import (
    ConditionalGaussian, ContinuousVAE, ContinuousVAEHparams, PaddedTensor, Perceiver,
    Transformer, TransformerHparams
)
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from typing import *
import torch


@dataclass
class TransformerVAEHparams(TransformerHparams, ContinuousVAEHparams):
    latent_depth: int = 64
    num_latent_vectors: int = 16


class TransformerVAE(Transformer, ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        self.example_input_array = None

        self.encoder_input_layer = deepcopy(self.input_layer)
        self.encoder_input_layer[0].weight = self.input_layer[0].weight

        self.encoder = Perceiver(
            num_layers=min(3, hparams.num_layers), num_latents=32, d_model=hparams.d_model,
            bottleneck_width=hparams.num_latent_vectors
        )
        self.q_of_z_given_x = ConditionalGaussian(hparams.num_latent_vectors * hparams.d_model, hparams.latent_depth)
        self.z_to_hidden = nn.Linear(hparams.latent_depth, hparams.d_model)

    def decoder_params(self) -> Iterable[nn.Parameter]:
        return chain(
            self.context_layer.parameters(),
            self.decoder_layers.parameters(),
            self.output_layer.parameters()
        )

    # Returns q(z|x)- only used for the console interface
    def encode(self, tokens: Tensor):
        x = self.input_layer(tokens)
        encoder_out = self.encoder(x)
        return self.q_of_z_given_x(encoder_out).flatten(-2)

    def training_step(self, batch: Dict[str, PaddedTensor], batch_index: int, stage: str = 'train'):
        original = batch['token_ids'].long()

        x = self.input_layer(original)
        encoder_out = self.encoder(x).flatten(-2)

        z, kl = self.sample_z(encoder_out, token_counts=batch['token_count'], stage=stage)
        logits = self.reconstruct(x, z)[..., :-1, :]
        nll = self.get_nll(logits, original[..., 1:], lengths=batch.get('num_char'), stage=stage)

        loss = (nll + self.hparams.kl_weight * kl)
        if stage == 'train':
            return loss
        elif stage == 'val':
            self.log('val_loss', nll + kl)

    def validation_step(self, batch: Dict[str, PaddedTensor], batch_index: int):
        return self.training_step(batch, batch_index, stage='val')

    def test_step(self, batch: Dict[str, PaddedTensor], batch_index: int):
        original = batch['token_ids']
        x = self.input_layer(original)

        posterior = self.q_of_z_given_x(self.encoder(x).flatten(-2))
        log_prob = self.estimate_log_prob_iw(posterior, x, original, num_samples=100, num_iter=20) / batch['token_count']
        nll_iw = -log_prob.mean()
        self.log('nll_iw', nll_iw, on_step=True)
        return nll_iw

    def reconstruct(self, x, z) -> Tensor:
        z_hidden = self.z_to_hidden(z).unsqueeze(-2)

        # Broadcast x across multiple samples of z if necessary
        if z.shape[0] > x.shape[0]:
            x = x.expand(z.shape[0], *x.shape[1:])

        for layer in self.decoder_layers:
            x = torch.cat([x[..., 0, None, :] + z_hidden, x[..., 1:, :]], dim=-2)
            x = layer(x) if not self.hparams.grad_checkpointing else checkpoint(layer, x)

        return self.output_layer(x)

    @torch.cuda.amp.autocast()
    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        # Unconditional samples will be mostly garbage when we haven't annealed to the full KL weight
        if self.hparams.kl_weight < 1.0:
            return None

        z = kwargs.get('z')
        if z is None:
            z = torch.randn(batch_size, self.hparams.num_latent_vectors, self.hparams.latent_depth, device=self.device)

        z = self.z_to_hidden(z)
        z = PaddedTensor.unpadded(z)
        return super().sample(max_length, batch_size, z=z, **kwargs)
