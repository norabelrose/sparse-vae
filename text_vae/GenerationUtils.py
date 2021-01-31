import torch
import torch.nn.functional as F
from enum import Enum
from torch import nn, Tensor
from typing import *


GenerationStrategy = Enum('GenerationStrategy', ['Beam', 'Greedy', 'Sampling'])

def autoregressive_decode(
    strategy: GenerationStrategy,
    rnn: nn.RNNBase,                    # Either an LSTM or a vanilla RNN
    z: Tensor,                          # Latent state. Either [batch, depth] or [batch, seq_len, depth]
    embedding: nn.Embedding,
    logit_callable: Callable,           # Upsamples from hidden states to logits
    initial_hidden_state: Tensor,
    start_symbol: int = 101,            # [CLS] in the tokenizer we're using
    end_symbol: int = 102,              # [SEP] in the tokenizer we're using
    min_length: int = 10,               # Will never emit [SEP] while the current length < min_length
    max_length: int = 200,
    k: int = 5                          # The beam size for beam search; ignored for other generation strategies
) -> Tensor:
    batch_size = z.size(0)
    if strategy == GenerationStrategy.Beam:
        log_probs = z.new_zeros([batch_size, k])    # It's 100% likely that a sample will start with [CLS]
        output_tensor = z.new_zeros([batch_size * k, max_length], dtype=torch.int32)
    else:
        output_tensor = z.new_zeros([batch_size, max_length], dtype=torch.int32)

    # Every sample starts with [CLS]
    output_tensor[:, 0] = start_symbol

    # [batch, (seq_len,)? depth]
    latent_has_length = (z.ndim() == 3)
    if latent_has_length:
        # Don't allow generation past the sequence length dimension of the latent tensor
        if max_length > z.shape[1]:
            max_length = z.shape[1]

    initial_hidden_state.unsqueeze_(0)  # nn.RNNBase expects a num_layers dimension at index 0
    h_init = torch.tanh(initial_hidden_state)

    # Vanilla RNNs don't have separate hidden states and cell states
    decoder_hidden = (h_init, initial_hidden_state) if isinstance(rnn, nn.LSTM) else h_init

    word_ids = output_tensor.new_full(output_tensor.shape[0], start_symbol)  # (batch_size) or (batch_size * k)
    end_symbol = word_ids.new_tensor(end_symbol)
    live_sample_mask = torch.ones_like(word_ids, dtype=torch.bool)           # (batch_size) or (batch_size * k)

    for current_idx in range(1, max_length):
        # Concatenate the word embedding of the previous token with the latent state to get the RNN input
        prev_word_embedding = embedding(word_ids)
        latent_to_concat = z if not latent_has_length else z[:, current_idx, :]
        rnn_input = torch.cat([prev_word_embedding, latent_to_concat], dim=-1).unsqueeze(1)

        output, decoder_hidden = rnn(rnn_input, decoder_hidden)
        output_logits = logit_callable(output)

        # Make the end symbol infinitely unlikely if we're not at the min length yet
        if current_idx < min_length:
            output_logits[:, :, end_symbol] = -float('inf')

        if strategy == GenerationStrategy.Beam:
            # (batch_size * k, 1, vocab_size) -> (batch_size, k, vocab_size)
            output_logits = output_logits.view(batch_size, k, -1)
            logits, word_ids = output_logits.topk(k=k, sorted=False)          # (batch_size, k, k)
            hypothesis_log_probs = F.log_softmax(logits, dim=-1)              # (batch_size, k, k)
            hypothesis_log_probs += log_probs.unsqueeze(-1)  # noqa; joint probs of the hypotheses plus these new words
            hypothesis_log_probs = hypothesis_log_probs.view(batch_size, -1)  # (batch_size, k^2)

            log_probs, indices_to_keep = hypothesis_log_probs.topk(k=k, sorted=False)  # (batch_size, k)
            word_ids = word_ids[indices_to_keep].flatten()  # (batch_size * k)

        elif strategy == GenerationStrategy.Greedy:
            word_ids = output_logits.argmax(dim=1)  # (batch_size)

        elif strategy == GenerationStrategy.Sampling:
            word_ids = F.softmax(output_logits, dim=1).multinomial(num_samples=1).squeeze(1)  # (batch_size)

        word_ids[~live_sample_mask].zero_()  # After the [SEP] token, everything afterward should be the padding token
        output_tensor[:, current_idx] = word_ids

        live_sample_mask &= word_ids.neq(end_symbol)
        if not live_sample_mask.any():  # All samples in the batch have produced the end symbol
            # If we're doing beam search, get rid of all the sub-optimal hypotheses
            if strategy == GenerationStrategy.Beam:
                best_indices = log_probs.argmax(dim=-1)  # noqa

                output_tensor = output_tensor.view(batch_size, k, -1)  # (batch_size * k, len) ->(batch_size, k, len)
                output_tensor = output_tensor[best_indices]  # (batch_size, len)

            output_tensor = output_tensor[:, :current_idx + 1]  # Get rid of any excess padding
            break

    return output_tensor

def nonautoregressive_decode(
    logits: Tensor,
    min_length: int = 10,
    start_symbol: int = 101,
    special_token_limit: int = 1000,
    k: int = 1
) -> Tensor:
    # Make special tokens other than [CLS] infinitely unlikely until we hit the min length
    logits[:, :min_length, :special_token_limit] = -float('inf')
    logits[:, 0, start_symbol] = float('inf')  # [CLS] is infinitely likely at the beginning of the sentence

    output = logits.topk(k).indices
    return output.squeeze(-1) if k == 1 else output
