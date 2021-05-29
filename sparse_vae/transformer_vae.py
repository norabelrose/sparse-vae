from .core import (
    Attention, GenerationState,
    ConditionalGaussian, ContinuousVAE, ContinuousVAEHparams, PaddedTensor, Perceiver,
    Transformer, TransformerHparams, custom_gaussian_rbf_mmd_sq, mutual_info_monte_carlo
)
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from typing import *
import torch
import torch.nn.functional as F


@dataclass
class TransformerVAEHparams(TransformerHparams, ContinuousVAEHparams):
    latent_depth: int = 64
    mutual_info_mean: float = 100.0
    mutual_info_sd: float = 4.0
    mmd: bool = False
    early_stopping_metric: str = 'val_nll'


class TransformerVAE(Transformer, ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        self.example_input_array = None

        self.encoder_input_layer = deepcopy(self.input_layer)
        self.encoder_input_layer[0].weight = self.input_layer[0].weight

        if self.hparams.mmd:
            self.bottleneck = nn.Linear(hparams.d_model, hparams.latent_depth)

            # The Langrangian multiplier is optimized "adversarially" to maximize the loss.
            # To prevent it from diverging to infinity when the MMD isn't exactly zero, we
            # use a "slack" of 0.05 and decay the multiplier when the MMD is within the slack
            self.lambda_mmd = nn.Parameter(torch.tensor(0.01))
            self.lambda_mmd.register_hook(lambda grad: -grad + 3.0)
        else:
            self.q_of_z_given_x = ConditionalGaussian(hparams.d_model, hparams.latent_depth)

        self.encoder = Perceiver(
            num_layers=min(3, hparams.num_layers), num_latents=64, d_model=hparams.d_model, bottleneck_width=1
        )
        self.z_projections = nn.ModuleList([
            nn.Linear(hparams.latent_depth, hparams.d_model)
            for _ in range(hparams.num_layers)
        ])

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
        return self.q_of_z_given_x(encoder_out)

    def training_step(self, batch: Dict[str, PaddedTensor], batch_index: int, stage: str = 'train'):
        original = batch['token_ids'].long()
        if original.shape[0] <= 1:
            return None

        x = self.input_layer(original)
        encoder_out = self.encoder(x)

        if self.hparams.mmd:
            var = self.prior_logvar.exp()
            z = self.bottleneck(encoder_out)
            z = F.layer_norm(z, [z.shape[-1]], var.sqrt(), self.prior_mean)

            # z = self.code_norm(z.squeeze(-2)).unsqueeze(-2)
            # z = self.code_norm(z)
            mmd = custom_gaussian_rbf_mmd_sq(z.view(z.shape[0], -1), self.prior_mean, var)
            # mmd = # analytic_gaussian_rbf_mmd_sq(z.view(z.shape[0], -1))

            logits = self.reconstruct(x, z)[..., :-1, :]
            nll = self.get_nll(logits, original[..., 1:], stage=stage)

            loss = nll + self.lambda_mmd * mmd  # torch.clamp(mmd - 2.0, 0.0)
            self.log('lambda_mmd', self.lambda_mmd.data)
            self.log(stage + '_mmd', mmd)

            var, mean = torch.var_mean(z, dim=0)
            residuals = z - mean

            self.log('z_mean', mean.mean())
            self.log('z_var', var.mean())
            self.log('z_skew', (residuals / var.sqrt()).pow(3.0).mean())
            self.log('z_kurtosis', (residuals.pow(4.0) / var.pow(2.0)).mean())

            if stage == 'train':
                return {'loss': loss, 'z': z}
            elif stage == 'val':
                self.log('val_loss', loss)
        else:
            z, kl, posterior = self.sample_z(encoder_out, token_counts=batch['token_count'], stage=stage)

            logits = self.reconstruct(x, z)[..., :-1, :]
            nll = self.get_nll(logits, original[..., 1:], byte_counts=batch['num_bytes'], stage=stage)
            loss = nll + self.hparams.kl_weight * kl

            mutual_info = mutual_info_monte_carlo(posterior)
            # self.log(stage + '_mutual_info', mutual_info)
            self.log(stage + '_mc_mutual_info', mutual_info)
            if mi_mode := self.hparams.mutual_info_mean:
                loss = loss + 0.5 * (mutual_info - mi_mode) ** 2 / (x.shape[-2] * self.hparams.mutual_info_sd ** 2)

            if stage == 'train':
                return {'loss': loss, 'posterior': posterior}
            elif stage == 'val':
                self.log('val_loss', nll + kl)

    def validation_step(self, batch: Dict[str, PaddedTensor], batch_index: int):
        return self.training_step(batch, batch_index, stage='val')

    def test_step(self, batch: Dict[str, PaddedTensor], batch_index: int):
        original = batch['token_ids']
        x = self.input_layer(original)

        posterior = self.q_of_z_given_x(self.encoder(x).squeeze(-2))
        log_prob = self.estimate_log_prob_iw(posterior, x, original, num_samples=100, num_iter=20) / batch['token_count']
        nll_iw = -log_prob.mean()
        self.log('nll_iw', nll_iw, on_step=True)
        return nll_iw

    def reconstruct(self, x, z) -> Tensor:
        should_checkpoint = self.hparams.grad_checkpointing and x.requires_grad
        # z = self.z_to_hidden(z)
        for i, layer in enumerate(self.decoder_layers):
            z_hidden = self.z_projections[i](z)
            x = torch.cat([z_hidden, x[..., 1:, :]], dim=-2)
            x = layer(x) if not should_checkpoint else checkpoint(layer, x)

        return self.output_layer(x)

    @torch.cuda.amp.autocast()
    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        # Unconditional samples will be mostly garbage when we haven't annealed to the full KL weight
        if self.hparams.kl_weight < 1.0:
            return None

        z = kwargs.pop('z', None)
        if z is None:
            z = torch.randn(batch_size, 1, self.hparams.latent_depth, device=self.device)

        # z = self.z_to_hidden(z)
        state = GenerationState(
            max_length, batch_size, self.start_token, self.end_token, device=self.device, **kwargs
        )
        state.output_ids[:, 0] = self.start_token

        with Attention.kv_cache(max_length):
            while not state.should_stop():
                inputs = state.prev_tokens()
                x = self.input_layer(inputs)

                for i, layer in enumerate(self.decoder_layers):
                    if state.current_index == 1:
                        x = torch.cat([self.z_projections[i](z), x[..., 1:, :]], dim=-2)

                    x = layer(x)

                next_logits = self.output_layer(x.squeeze(1))
                continuing_mask = state.process_logits(next_logits)

                Attention.update_kv_cache(continuing_mask)

        return state.final_output()
