from enum import Enum
from torch import nn, Tensor
from typing import *
import torch


GenerationStrategy = Enum('GenerationStrategy', ['Beam', 'Greedy', 'Sampling', 'SamplingTopK', 'SamplingTopP'])

@torch.no_grad()
def autoregressive_decode(
    strategy: GenerationStrategy,
    rnn: Callable,                      # An LSTM or a vanilla RNN
    z: Optional[Tensor],                # Latent state. Either [batch, depth] or [batch, seq_len, depth]
    embedding: nn.Embedding,
    logit_callable: Callable,           # Upsamples from hidden states to logits
    initial_hidden_state: Tensor,
    start_symbol: int = 101,            # [CLS] in the tokenizer we're using
    end_symbol: int = 102,              # [SEP] in the tokenizer we're using
    min_length: int = 10,               # Will never emit [SEP] while the current length < min_length
    max_length: int = 200,
    k: int = 5,                         # The beam size for beam search and top K sampling

    num_samples: int = 1
) -> Tensor:
    device = embedding.weight.data.device
    if z is not None:
        num_samples = z.shape[0]

    if strategy == GenerationStrategy.Beam:
        beam_log_probs = torch.zeros(num_samples, k, device=device)  # 100% likely that a sample will start with [CLS]
        output_tensor = torch.zeros(num_samples * k, max_length, device=device, dtype=torch.long)
    else:
        beam_log_probs = None
        output_tensor = torch.zeros(num_samples, max_length, device=device, dtype=torch.long)

    # Every sample starts with [CLS]
    output_tensor[:, 0] = start_symbol

    # [batch, (seq_len,)? depth]
    latent_has_length = (z.ndim == 3) if z is not None else False
    if latent_has_length:
        # Don't allow generation past the sequence length dimension of the latent tensor
        if max_length > z.shape[1]:
            max_length = z.shape[1]

    h_init = initial_hidden_state.tanh()

    # Vanilla RNNs don't have separate hidden states and cell states
    decoder_hidden = (h_init, initial_hidden_state) if isinstance(rnn, nn.LSTM) else h_init

    # (batch_size, 1) or (batch_size * k, 1)
    word_ids = torch.full((output_tensor.shape[0], 1), start_symbol, device=device)
    end_symbol = torch.tensor(end_symbol, device=device)
    live_sample_mask = torch.ones_like(word_ids, dtype=torch.bool)           # (batch_size, 1) or (batch_size * k, 1)

    for current_idx in range(1, max_length):
        # Concatenate the word embedding of the previous token with the latent state to get the RNN input
        prev_word_embedding = embedding(word_ids)
        if z is not None:
            latent_to_concat = z if not latent_has_length else z[:, current_idx, :]
            rnn_input = torch.cat([prev_word_embedding, latent_to_concat.unsqueeze(1)], dim=-1)
        else:
            rnn_input = prev_word_embedding

        output, decoder_hidden = rnn(rnn_input, decoder_hidden)
        output_logits = logit_callable(output)

        # Make the end symbol infinitely unlikely if we're not at the min length yet
        if current_idx < min_length:
            output_logits[:, :, end_symbol] = -float('inf')

        word_ids = decode_next_token_from_logits(output_logits.squeeze(1), strategy, k=k,
                                                 beam_log_probs=beam_log_probs).unsqueeze(-1)

        word_ids[~live_sample_mask].zero_()  # After the [SEP] token, everything afterward should be the padding token
        output_tensor[:, current_idx] = word_ids

        live_sample_mask &= word_ids.ne(end_symbol).squeeze(-1)
        if not live_sample_mask.any():  # All samples in the batch have produced the end symbol
            # If we're doing beam search, get rid of all the sub-optimal hypotheses
            if strategy == GenerationStrategy.Beam:
                best_indices = beam_log_probs.argmax(dim=-1)  # noqa

                output_tensor = output_tensor.view(num_samples, k, -1)  # (batch_size * k, len) -> (batch_size, k, len)
                output_tensor = output_tensor[:, best_indices]  # (batch_size, len)

            output_tensor = output_tensor[:, :current_idx + 1]  # Get rid of any excess padding
            break

    return output_tensor

@torch.no_grad()
def decode_next_token_from_logits(logits: Tensor, strategy: GenerationStrategy, k: int = 10, p: float = 0.92,
                                  beam_log_probs = None):
    if strategy == GenerationStrategy.Beam:
        num_samples = logits.shape[0] // k

        # (batch_size * k, 1, vocab_size) -> (batch_size, k, vocab_size)
        output_logits = logits.view(num_samples, k, -1)
        logits, token_ids = output_logits.topk(k=k, sorted=False)  # (batch_size, k, k)
        hypothesis_log_probs = logits.log_softmax(dim=-1)  # (batch_size, k, k)
        hypothesis_log_probs += beam_log_probs.unsqueeze(-1)  # noqa; joint probs of the hypotheses plus these new words
        hypothesis_log_probs = hypothesis_log_probs.view(num_samples, -1)  # (batch_size, k^2)

        indices_to_keep = torch.empty_like(beam_log_probs, dtype=torch.long)
        torch.topk(hypothesis_log_probs, k=k, sorted=False, out=(beam_log_probs, indices_to_keep))  # (batch_size, k))
        return token_ids.flatten()[indices_to_keep].squeeze()  # (batch_size * k)

    elif strategy == GenerationStrategy.Greedy:
        return logits.argmax(dim=-1)  # (batch_size)

    elif strategy == GenerationStrategy.SamplingTopK:
        top_logits, indices = logits.topk(k=k)

        dist = top_logits.softmax(dim=-1).flatten(end_dim=-2)
        subindices = dist.multinomial(num_samples=1).view(*top_logits.shape[:-1], 1)
        return indices.gather(dim=-1, index=subindices).squeeze(-1)

    elif strategy == GenerationStrategy.SamplingTopP:
        sorted_logits, indices = logits.sort(descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cum_probs = sorted_probs.cumsum(dim=-1)

        unlikely_token_mask = (cum_probs > p)
        unlikely_token_mask[..., 0] = False     # Always exclude the very most probable token from being removed
        sorted_probs[unlikely_token_mask] = 0.0

        subindices = sorted_probs.multinomial(num_samples=1).view(*sorted_logits.shape[:-1], 1)
        return indices.gather(dim=-1, index=subindices).squeeze(-1)

    else:  # Regular sampling
        dist = logits.softmax(dim=-1).flatten(end_dim=-2)
        return dist.multinomial(num_samples=1).view(*logits.shape[:-1])
