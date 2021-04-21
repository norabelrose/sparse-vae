import torch
from dataclasses import dataclass
from itertools import chain
from omegaconf import DictConfig
from .core import (
    ContinuousVAE, ConditionalGaussian, ContinuousVAEHparams, GenerationState, Perceiver
)
from torch import nn, Tensor
from typing import *


@dataclass
class LSTMVAEHparams(ContinuousVAEHparams):
    latent_depth: int = 32
    num_latent_vectors: int = 1

    bidirectional_encoder: bool = False
    transformer_encoder: bool = False
    decoder_input_dropout: float = 0.0      # Should make decoder pay more attention to the latents
    decoder_output_dropout: float = 0.0     # Sort of like label smoothing?

    divide_loss_by_length: bool = True
    tie_embedding_weights: bool = True  # Tie the decoder's embedding weights to the encoder's embedding weights

    d_embedding: int = 512  # Dimensionality of the input embedding vectors
    d_model: int = 1024  # Dimensionality of the LSTM hidden state
    num_layers: int = 1
    tie_logit_weights: bool = False  # Tie the logit layer weights to the embedding weights


class LSTMVAE(ContinuousVAE):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        self.example_input_array = None

        self.encoder_embedding = nn.Embedding(hparams.vocab_size, hparams.d_embedding)
        self.decoder_embedding = nn.Embedding(hparams.vocab_size, hparams.d_embedding)
        if hparams.tie_embedding_weights:
            self.encoder_embedding.weight = self.decoder_embedding.weight

        self.decoder = nn.LSTM(
            input_size=hparams.d_embedding + hparams.latent_depth,
            hidden_size=hparams.d_model,
            batch_first=True,
            num_layers=hparams.num_layers
        )
        self.c0 = nn.Parameter(torch.randn(hparams.num_layers, 1, hparams.d_model))

        if hparams.tie_logit_weights:
            output_embedding = nn.Linear(hparams.d_embedding, hparams.vocab_size)
            output_embedding.weight = self.decoder_embedding.weight

            self.output_layer = nn.Sequential(
                # Note that we have to include this bottleneck here in order to tie the input and output embeddings
                nn.Linear(hparams.d_model, hparams.d_embedding),
                output_embedding
            )
        else:
            self.output_layer = nn.Linear(hparams.d_model, hparams.vocab_size)

        if hparams.transformer_encoder:
            hidden_size = hparams.d_embedding
            self.encoder = Perceiver(
                num_layers=3, num_latents=32, d_model=hparams.d_embedding, bottleneck_width=hparams.num_latent_vectors
            )
        else:
            self.encoder = nn.LSTM(
                input_size=hparams.d_embedding,
                hidden_size=hparams.d_model,
                bidirectional=hparams.bidirectional_encoder,
                num_layers=hparams.num_layers,
                batch_first=True
            )
            num_directions = 2 if hparams.bidirectional_encoder else 1
            hidden_size = hparams.d_model * num_directions
            self.c0 = nn.Parameter(torch.randn(num_directions, 1, hparams.d_model))

        self.automatic_optimization = hparams.train_mc_samples == 0

        # This is the posterior distribution when we're using the traditional VAE training objective
        # (i.e. mc_samples == 0), and the proposal distribution when using DReG (i.e. mc_samples > 0)
        self.q_of_z_given_x = ConditionalGaussian(hidden_size, hparams.latent_depth)
        self.z_to_hidden = nn.Linear(hparams.latent_depth, hparams.d_model)

        self.dropout_in = nn.Dropout(hparams.decoder_input_dropout)
        self.dropout_out = nn.Dropout(hparams.decoder_output_dropout)

        self.initialize_weights()

    def decoder_params(self) -> Iterable[nn.Parameter]:
        return chain(
            self.decoder.parameters(),
            self.decoder_embedding.parameters(),
            self.output_layer.parameters()
        )

    def forward(self, x):
        if self.hparams.transformer_encoder:
            return self.encoder(x).squeeze(-2)

        batch_size = x.shape[0]
        c0 = self.c0.repeat(1, batch_size, 1)
        _, (last_state, _) = self.encoder(x, (c0.tanh(), c0))

        if not self.hparams.transformer_encoder and last_state.shape[0] > 1:
            last_state = last_state.movedim(0, 1).flatten(1)
        else:
            last_state = last_state.squeeze(0)

        return last_state

    def training_step(self, batch: Dict[str, Tensor], batch_index: int, stage: str = 'train'):
        original = batch['token_ids']

        x = self.encoder_embedding(original)
        last_state = self.forward(x)

        if self.hparams.train_mc_samples > 0:
            q_of_z = self.q_of_z_given_x(last_state, get_kl=False)
            self.dreg_backward_pass(q_of_z, x, original)

            optimizer = self.optimizers()
            optimizer.step(); optimizer.zero_grad()

        # Single-sample VAE objective
        else:
            z, kl = self.sample_z(last_state, token_counts=batch['token_count'], stage=stage)
            if not self.hparams.tie_embedding_weights:
                x = self.decoder_embedding(original)

            logits = self.reconstruct(self.dropout_in(x), z)[..., :-1, :]  # Remove [SEP]
            nll = self.get_nll(logits, batch['token_ids'][..., 1:], word_counts=batch['num_words'])

            loss = (nll + self.hparams.kl_weight * kl)
            if stage == 'train':
                return {'logits': logits, 'loss': loss}
            elif stage == 'val':
                self.log('val_loss', nll + kl)

    def validation_step(self, batch: Dict[str, Tensor], batch_index: int):
        return self.training_step(batch, batch_index, stage='val')

    def test_step(self, batch: Dict[str, Tensor], batch_index: int):
        original = batch['token_ids']
        x = self.encoder_embedding(original)

        posterior = self.q_of_z_given_x(self.forward(x))
        log_prob = self.estimate_log_prob_iw(posterior, x, original, num_samples=100, num_iter=20) / batch[
            'token_count']
        nll_iw = -log_prob.mean()
        self.log('nll_iw', nll_iw, on_step=True)
        return nll_iw

    # x should be a batch of sequences of token embeddings and z a batch of single latent vectors; both
    # tensors may have a leading num_samples dimension if we're using a multi-sample Monte Carlo objective.
    def reconstruct(self, x, z):
        batch_size, seq_len, _ = x.shape[-3:]

        # Broadcast x across multiple samples of z if necessary
        if z.shape[0] > x.shape[0]:
            x = x.expand(z.shape[0], *x.shape[1:])

        # (num_samples?), batch_size, seq_len, d_model
        x = self.dropout_in(x)

        # Expand z across the sequence length and then concatenate it to each token embedding
        x = torch.cat([x, z.unsqueeze(-2).expand(*x.shape[:-1], self.hparams.latent_depth)], dim=-1)

        # Merge the minibatch and the MC sample dimensions if needed; nn.LSTM doesn't support multiple batch dims
        z = z.flatten(end_dim=-2)
        c_init = self.z_to_hidden(z).unsqueeze(0)
        h_init = c_init.tanh()
        output, _ = self.decoder(x.flatten(end_dim=-3), (h_init, c_init))   # noqa

        output = output.view(*x.shape[:-1], output.shape[-1])  # Add the MC sample dim again if needed
        output = self.dropout_out(output)
        return self.output_layer(output)

    def sample(self, max_length: int, batch_size: int = 1, **kwargs):
        # Unconditional samples will be mostly garbage when we haven't annealed to the full KL weight
        if self.hparams.kl_weight < 1.0:
            return None

        z = torch.randn(batch_size, self.hparams.latent_depth, device=self.device)
        initial_state = self.z_to_hidden(z).unsqueeze(0)

        state = GenerationState(
            max_length, batch_size, device=self.device, start_token=self.start_token, end_token=self.end_token
        )
        h_init = initial_state.tanh()
        decoder_hidden = (h_init, initial_state)

        while not state.should_stop():
            # Concatenate the word embedding of the previous token with the latent state to get the RNN input
            prev_word_embedding = self.decoder_embedding(state.prev_tokens())
            live_latents = z[state.live_sample_mask, :]

            lstm_input = torch.cat([prev_word_embedding, live_latents.unsqueeze(1)], dim=-1)
            output, decoder_hidden = self.decoder(lstm_input, decoder_hidden)
            logits = self.output_layer(output)

            continuing_mask = state.process_logits(logits.squeeze(1))
            decoder_hidden = tuple(x[..., continuing_mask, :] for x in decoder_hidden)

        return state.final_output()

    def context_depth(self) -> int:
        return self.hparams.latent_depth
