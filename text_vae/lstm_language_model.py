from dataclasses import dataclass
from omegaconf import DictConfig
from torch import nn, Tensor
from typing import *
import torch
from .core import GenerationState, LanguageModel, LanguageModelHparams


@dataclass
class LSTMLanguageModelHparams(LanguageModelHparams):
    d_embedding: int = 512  # Dimensionality of the input embedding vectors
    d_model: int = 1024  # Dimensionality of the LSTM hidden state
    num_layers: int = 1

    tie_logit_weights: bool = False  # Tie the logit layer weights to the embedding weights


class LSTMLanguageModel(LanguageModel):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        self.decoder_embedding = nn.Embedding(hparams.vocab_size, hparams.d_embedding)
        self.decoder = nn.LSTM(
            input_size=hparams.d_embedding,
            hidden_size=hparams.d_model,
            batch_first=True,
            num_layers=hparams.num_layers
        )
        self.initial_state = nn.Parameter(torch.randn(hparams.num_layers, 1, hparams.d_model))

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

    # Just here as a subclass hook for LSTMAutoencoder, which will make this d_embedding + latent_depth
    def decoder_input_size(self) -> int:
        return self.hparams.d_embedding

    # Returns [batch, seq_len, vocab_size] tensor of logits
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        x = batch['token_ids']
        batch_size = x.shape[0]
        c0 = self.initial_state.repeat(1, batch_size, 1)
        h0 = c0.tanh()

        x = self.decoder_embedding(x)
        x, _ = self.decoder(x, (h0, c0))
        return self.output_layer(x)

    def sample(self, max_length: int, batch_size: int = 1, initial_state: Tensor = None, context = None, **kwargs):
        initial_state = initial_state if initial_state is not None else self.initial_state.repeat(1, batch_size, 1)

        state = GenerationState(
            max_length=max_length,
            batch_size=batch_size,
            beam_size=1,
            device=self.device,
            dtype=torch.long,
            end_token=self.end_token,
            top_p=0.92  # Nucleus sampling
        )
        state.output_ids[:, 0] = self.start_token  # Every sample starts with [CLS]

        h_init = initial_state.tanh()
        decoder_hidden = (h_init, initial_state)

        for current_idx in range(1, max_length):
            # Concatenate the word embedding of the previous token with the latent state to get the RNN input
            prev_word_embedding = self.decoder_embedding(state.prev_tokens())
            if context is not None:
                lstm_input = torch.cat([prev_word_embedding, context.unsqueeze(1)], dim=-1)
            else:
                lstm_input = prev_word_embedding

            output, decoder_hidden = self.decoder(lstm_input, decoder_hidden)
            logits = self.output_layer(output)

            state.process_logits(logits.squeeze(1))
            if state.should_stop():  # All samples in the batch have produced the end symbol
                break

        return state.final_output()
