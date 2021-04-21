from dataclasses import dataclass, InitVar
from torch import Tensor
from typing import Optional
import torch


@dataclass
class GenerationState:
    max_length: InitVar[int]
    batch_size: InitVar[int]
    start_token: int
    end_token: int
    device: InitVar[torch.device]
    dtype: InitVar[torch.dtype] = torch.long

    top_k: int = 0
    top_p: float = 0.9
    temperature: float = 1.0
    length_penalty: float = 1.0
    rolling: bool = False

    def __post_init__(self, max_length: int, batch_size: int, device: torch.device, dtype: torch.dtype):
        self.output_ids = torch.zeros(batch_size, max_length, device=device, dtype=dtype)
        self.output_ids[:, 0] = self.start_token

        self.live_sample_mask = torch.ones(batch_size, device=device, dtype=torch.bool) if not self.rolling else None
        self.current_index = 1 if not self.rolling else torch.ones(batch_size, device=device, dtype=torch.long)

    def prev_tokens(self) -> Tensor:
        if self.rolling:
            return self.output_ids.gather(dim=-1, index=self.current_index[:, None] - 1)

        return self.output_ids[self.live_sample_mask, self.current_index - 1, None]

    def process_logits(self, logits: Tensor):
        num_beams = 1
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
            token_ids = indices if token_ids is None else token_ids.gather(dim=-1, index=indices).squeeze(-1)

        token_ids = token_ids.flatten()  # (batch_size * num_beams)
        self.output_ids[self.live_sample_mask, self.current_index] = token_ids.type_as(self.output_ids)

        self.current_index += 1
        continuing_sample_mask = (token_ids != self.end_token) & (self.current_index < self.output_ids.shape[-1])
        self.live_sample_mask[self.live_sample_mask.clone()] &= continuing_sample_mask

        return continuing_sample_mask

    def should_stop(self) -> bool:
        assert not self.rolling
        return self.current_index >= self.output_ids.shape[-1] - 1 or not self.live_sample_mask.any()

    def final_output(self) -> Tensor:
        return self.output_ids[:, 1:]   # Don't include the start token

    def pop_samples(self, finished_sample_mask: Tensor) -> Optional[Tensor]:
        if not finished_sample_mask.any():
            return None

        samples = self.output_ids[finished_sample_mask, :self.current_index]
        self.output_ids = self.output_ids[self.live_sample_mask, :]
        self.live_sample_mask = self.live_sample_mask.new_ones([self.output_ids.shape[0]])

        return samples
