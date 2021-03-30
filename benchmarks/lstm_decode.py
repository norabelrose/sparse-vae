from text_vae import GenerationState
from torch import nn, Tensor
from typing import *
import torch


@torch.no_grad()
def autoregressive_decode(
        rnn: Callable,  # An LSTM or a vanilla RNN
        z: Optional[Tensor],  # Latent state. Either [batch, depth] or [batch, seq_len, depth]
        embedding: nn.Embedding,
        logit_callable: Callable,  # Upsamples from hidden states to logits
        initial_hidden_state: Tensor,
        start_symbol: int = 101,  # [CLS] in the tokenizer we're using
        end_symbol: int = 102,  # [SEP] in the tokenizer we're using
        max_length: int = 200,
        beam_size: int = 1,
        k: int = 0,  # The K for top K sampling

        num_samples: int = 1
) -> Tensor:
    device = embedding.weight.data.device
    if z is not None:
        num_samples = z.shape[0]

    state = GenerationState(max_length, num_samples, beam_size, device, torch.long)
    state.output_ids[:, 0] = start_symbol  # Every sample starts with [CLS]

    # [batch, (seq_len,)? depth]
    latent_has_length = (z.ndim == 3) if z is not None else False
    if latent_has_length:
        # Don't allow generation past the sequence length dimension of the latent tensor
        if max_length > z.shape[1]:
            max_length = z.shape[1]

    h_init = initial_hidden_state.tanh()

    # Vanilla RNNs don't have separate hidden states and cell states
    decoder_hidden = (h_init, initial_hidden_state) if isinstance(rnn, nn.LSTM) else h_init

    for current_idx in range(1, max_length):
        # Concatenate the word embedding of the previous token with the latent state to get the RNN input
        prev_word_embedding = embedding(state.prev_tokens())
        if z is not None:
            latent_to_concat = z if not latent_has_length else z[:, current_idx, :]
            rnn_input = torch.cat([prev_word_embedding, latent_to_concat.unsqueeze(1)], dim=-1)
        else:
            rnn_input = prev_word_embedding

        output, decoder_hidden = rnn(rnn_input, decoder_hidden)
        output_logits = logit_callable(output)

        state.process_logits(output_logits.squeeze(1), end_token=end_symbol, top_k=k, top_p=0.92)
        if state.should_stop():  # All samples in the batch have produced the end symbol
            break

    return state.final_output()
