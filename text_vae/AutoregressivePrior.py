from dataclasses import dataclass
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.distributions import Categorical, Normal, kl_divergence
from typing import *
import torch
import torch.nn.functional as F
from .core import (
    LanguageModel, TransformerLanguageModel, TransformerLanguageModelHparams, TransformerLayer,
    decode_next_token_with_context, get_positional_encodings
)

@dataclass
class AutoregressiveCategoricalPriorHparams(TransformerLanguageModelHparams):
    codebook_size: int = 512


class AutoregressiveCategoricalPrior(TransformerLanguageModel):
    def __init__(self, hparams: DictConfig):
        # The last two codes in the codebook are a start-of-sequence token, so that the model can learn to predict the
        # very first position of the sequence given that token, and a end-of-sequence token, so that the model can
        # predict an early end to the latent sequence.
        hparams.vocab_size = hparams.codebook_size + 2
        self.start_token = hparams.codebook_size
        self.end_token = hparams.codebook_size + 1

        super().__init__(hparams)

    def forward(self, batch: Dict[str, Tensor]) -> Categorical:
        ground_truth = batch['posteriors']
        ground_truth = F.pad(ground_truth, (1, 1), self.start_token)
        ground_truth[..., -1] = self.end_token

        codes = self.input_layer(ground_truth)
        logits = self.output_layer(self.decoder(codes))
        return Categorical(logits=logits)

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, val: bool = False) -> Tensor:
        predicted_codes = self.forward(batch)

        ground_truth = batch['posteriors']
        initial_log_probs = self.initial_prior.log_prob(ground_truth[..., 0])
        conditional_log_probs = predicted_codes.log_prob(ground_truth[..., 1:])
        overall_log_probs = initial_log_probs + conditional_log_probs

        loss = -overall_log_probs.mean()
        self.log('train_nll' if not val else 'val_nll', loss)
        return loss

    def sample(self, max_length: int, count: int = 1, start_embedding: Tensor = None, **kwargs):
        start_symbol = torch.tensor(start_symbol, device=self.device)
        stop_symbol = torch.tensor(end_symbol, device=self.device)

        live_sample_mask = torch.ones(count, device=self.device, dtype=torch.bool)

        out_embeds = get_positional_encodings(max_length, d_model, device, dtype=torch.float32)
        out_embeds = out_embeds.unsqueeze(0).expand(count, *out_embeds.shape)
        out_embeds[:, 0] += start_embedding if start_embedding is not None else embedding(start_symbol)

        output_ids = torch.zeros(count, max_length, device=self.device, dtype=torch.long)
        output_ids[:, 0] = start_symbol

        assert strategy != GenerationStrategy.Beam

        for current_idx in range(1, max_length):
            ctx = context[:, :current_idx] if context is not None else None
            next_output = transformer_callable(out_embeds[:, :current_idx], ctx)[:, -1]
            next_logits = logit_callable(next_output)

            # Make the end symbol infinitely unlikely if we're not at the min length yet
            if current_idx < min_length:
                next_logits[:, end_symbol] = -float('inf')

            next_ids = decode_next_token_with_context(next_logits, strategy, k)

            output_ids[:, current_idx] = next_ids
            out_embeds[:, current_idx] += embedding(next_ids)

            live_sample_mask &= (next_ids != stop_symbol)
            if not live_sample_mask.any():
                output_ids = output_ids[:, :current_idx + 1]  # Get rid of any excess padding
                break

        return output_ids


class AutoregressiveGaussianPrior(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        self.initial_mu = nn.Parameter(torch.zeros(hparams.d_model))
        self.initial_logsigma = nn.Parameter(torch.zeros(hparams.d_model))

        self.transformer = nn.Sequential(*[
            TransformerLayer(d_model=hparams.d_model, num_heads=hparams.num_heads, causal=True)
            for _ in range(hparams.num_layers)
        ])

    # Given a Tensor of disentangled Gaussian parameters, predict the parameters for position N from the parameters at
    # all positions < N.
    def forward(self, batch: Dict[str, Tensor]) -> Normal:
        ground_truth = batch['posteriors']

        decoder_input = self.interleave_parameters(ground_truth)
        predicted_params = self.decoder(decoder_input)
        predicted_params = self.disentangle_parameters(predicted_params)
        return self.parameters_to_distribution(predicted_params)

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, val: bool = False) -> Tensor:
        predicted_dists = self.forward(batch)

        ground_truth_params = batch['posteriors']
        ground_truth_initial = self.parameters_to_distribution(ground_truth_params[..., 0, :])
        ground_truth_conditional = self.parameters_to_distribution(ground_truth_params[..., 1:, :])

        # Unlike a normal language model in which the model is never asked to predict the very first token- which is
        # generally just a filler token like [CLS]- here we actually do want the model to be able to predict the very
        # first position. We just learn a fixed Gaussian which should end up being the 'average' distribution we see
        # in the first position of the input samples.
        initial_kl = kl_divergence(ground_truth_initial, self.initial_prior)
        conditional_kl = kl_divergence(ground_truth_conditional, predicted_dists)
        overall_kl = initial_kl + conditional_kl

        # Loss is the average *per position* KL divergence between our predicted distributions and the actual ones
        loss = overall_kl.sum(dim=-1).mean()
        self.log('train_kl' if not val else 'val_kl', loss)
        return loss

    def sample(self, max_length: int, count: int = 1, **kwargs):
        pass

    # We have to recreate this object every time since we want to learn the scale parameter in log space (so that
    # negative values are actually valid), but you can't create a Normal object with a log sigma parameter
    @property
    def initial_prior(self) -> Normal:
        return Normal(loc=self.initial_mu, scale=self.initial_logsigma)

    @staticmethod
    def parameters_to_distribution(params: Tensor) -> Normal:
        mus, logsigmas = params.chunk(2, dim=-1)
        return Normal(loc=mus, scale=logsigmas.exp())

    # In order to store the mu and logsigma parameters of a diagonal Gaussian in a single Tensor, we actually use an
    # 'interleaved' format in which the mu and logsigma parameters for each latent variable are stored side-by-side
    # along the last dimension of the Tensor, instead of the common and convenient format of simply concatenating mu
    # and logsigma Tensors together along the final dimension. This is so that each attention head in the multi-head
    # attention modules can see both the mu and sigma parameters for the latent variables that it is attending over.
    # Otherwise, half of the attention heads would only see mu parameters and the other half would only see logsigma
    # parameters, which seems suboptimal- related pieces of information would be processed separately.
    @staticmethod
    def interleave_parameters(params: Tensor) -> Tensor:
        interleaved = torch.empty_like(params)
        mus, logsigmas = interleaved.chunk(2, dim=-1)

        interleaved[..., ::2] = mus
        interleaved[..., 1::2] = logsigmas
        return interleaved

    @staticmethod
    def disentangle_parameters(params: Tensor) -> Tensor:
        disentangled = torch.empty_like(params)
        mus, logsigmas = disentangled.chunk(2, dim=-1)
        mus.copy_(params[..., ::2])
        logsigmas.copy_(params[..., 1::2])
        return disentangled
