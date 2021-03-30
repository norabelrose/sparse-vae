from dataclasses import dataclass, InitVar
from torch import Tensor
import torch


@dataclass
class GenerationState:
    max_length: InitVar[int]
    batch_size: InitVar[int]
    beam_size: InitVar[int]
    device: InitVar[torch.device]
    dtype: InitVar[torch.dtype]
    end_token: int

    top_k: int = 0
    top_p: float = 1.0
    temperature: float = 1.0
    length_penalty: float = 1.0

    def __post_init__(self, max_length: int, batch_size: int, beam_size: int, device: torch.device, dtype: torch.dtype):
        num_samples = batch_size * beam_size
        self.beam_log_probs = torch.zeros(batch_size, beam_size, device=device) if beam_size > 1 else None
        self.beam_lengths = self.beam_log_probs.clone() if beam_size > 1 else None

        self.output_ids = torch.zeros(num_samples, max_length, device=device, dtype=dtype)
        self.live_sample_mask = torch.ones(num_samples, device=device, dtype=torch.bool)
        self.current_index = 1

    def prev_tokens(self) -> Tensor:
        return self.output_ids[self.live_sample_mask, self.current_index - 1, None]

    def process_logits(self, logits: Tensor):
        num_beams = self.beam_log_probs.shape[1] if self.beam_log_probs is not None else 1
        token_log_probs = logits.log_softmax(dim=-1) if num_beams > 1 else None
        token_ids = None

        # Zero temperature means argmax/topk
        if self.temperature <= 0.0 or self.top_k == 1:
            logits, token_ids = logits.topk(k=num_beams, sorted=False)

        # All strategies other than greedy search- some randomness is involved
        else:
            logits /= self.temperature

            # Top K sampling
            if self.top_k > 0:
                self.top_k = max(self.top_k, num_beams)  # Since we sample without replacement w/ beam search, k must >= num_beams
                logits, token_ids = logits.topk(k=self.top_k, sorted=False)

            # Sample from a truncated distribution containing only the most probable tokens whose cumulative probability
            # is no greater than some parameter P (the 'nucleus')
            if self.top_p < 1.0:
                logits, token_ids = logits.sort(descending=True)
                probs = logits.softmax(dim=-1)
                cum_probs = probs.cumsum(dim=-1)

                unlikely_token_mask = (cum_probs > self.top_p)
                unlikely_token_mask[..., :num_beams] = False  # Exclude the very most probable token from being removed
                probs[unlikely_token_mask] = 0.0
            else:
                probs = logits.softmax(dim=-1)

            # Note that in the beam search case we're sampling *without replacement* to ensure that
            # we explore other possibilities and we don't waste computation
            indices = probs.multinomial(num_samples=num_beams).view(*logits.shape[:-1], num_beams)
            if token_ids is None:
                token_ids = indices
            else:
                token_ids = token_ids.gather(dim=-1, index=indices).squeeze(-1)

        # Any of the above sampling strategies can be combined with beam search
        # in order to trade off between exploration and exploitation
        if num_beams > 1:
            num_live_samples = self.live_sample_mask.sum()
            num_dead_samples = self.live_sample_mask.numel() - num_live_samples
            num_hypotheses = num_dead_samples + num_live_samples * num_beams

            new_log_probs = self.beam_log_probs.new_empty(num_hypotheses)
            new_lengths = new_log_probs.new_empty(num_hypotheses)

            new_log_probs[self.live_sample_mask] += token_log_probs.gather(dim=-1, index=token_ids)
            new_lengths[self.live_sample_mask] += (token_ids != self.end_token)

            # We now have num_beams ** 2 possibilities, let's group them in one dimension so we can use
            # topk to weed out the lowest scored options
            new_log_probs = new_log_probs.view(-1, num_beams ** 2)
            new_lengths = new_lengths.view(-1, num_beams ** 2)

            # Longer sequences are penalized if the penalty > 1; they are encouraged if penalty < 1
            if self.length_penalty != 1.0:
                new_lengths **= self.length_penalty

            scores = (new_log_probs / new_lengths)
            indices_to_keep = scores.topk(k=num_beams, sorted=False).indices

            self.beam_log_probs = new_log_probs.gather(dim=-1, index=indices_to_keep)
            self.beam_lengths = new_lengths.gather(dim=-1, index=indices_to_keep)
            token_ids = token_ids.gather(dim=-1, index=indices_to_keep)
        else:
            token_ids = token_ids.flatten()  # (batch_size * num_beams)
            self.output_ids[self.live_sample_mask, self.current_index] = token_ids.type_as(self.output_ids)

        self.live_sample_mask[self.live_sample_mask.clone()] &= (token_ids != self.end_token)
        self.current_index += 1

    def should_stop(self) -> bool:
        return self.current_index >= self.output_ids.shape[-1] - 1 or not self.live_sample_mask.any()

    def final_output(self) -> Tensor:
        # If we're doing beam search, get rid of all the sub-optimal hypotheses
        if self.beam_log_probs is not None:
            best_indices = (self.beam_log_probs / self.beam_lengths).argmax(dim=-1)

            output_ids = self.output_ids.view(*self.beam_log_probs.shape, -1)  # (batch_size * k, len) -> (batch_size, k, len)
            return output_ids[:, best_indices]  # (batch_size, len)

        return self.output_ids
