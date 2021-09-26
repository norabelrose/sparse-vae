from dataclasses import dataclass, InitVar
from torch import Tensor
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
    repetition_penalty: float = 1.2

    def __post_init__(self, max_length: int, batch_size: int, device: torch.device, dtype: torch.dtype):
        self.output_ids = torch.zeros(batch_size, max_length, device=device, dtype=dtype)
        self.output_ids[:, 0] = self.start_token

        self.live_sample_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
        self.current_index = 1

    def prev_tokens(self) -> Tensor:
        return self.output_ids[self.live_sample_mask, self.current_index - 1, None]

    def process_logits(self, logits: Tensor):
        token_ids = None

        # Penalize the logits of any tokens we've already generated to avoid repetition
        if (penalty := self.repetition_penalty) > 1.0:
            left_start = max(self.current_index - 512, 0)   # Ignore things we generated more than 512 timesteps ago
            prev_tokens = self.output_ids[self.live_sample_mask, left_start:self.current_index]
            prev_logits = logits.gather(dim=-1, index=prev_tokens)
            penalized_logits = torch.where(prev_logits < 0.0, prev_logits * penalty, prev_logits / penalty)
            logits.scatter_(dim=-1, index=prev_tokens, src=penalized_logits)

        # Zero temperature means argmax/topk
        if self.temperature <= 0.0 or self.top_k == 1:
            logits, token_ids = logits.max(dim=-1, keepdim=True)

        # All strategies other than greedy search- some randomness is involved
        else:
            logits /= self.temperature

            # Top K sampling
            if self.top_k > 0:
                self.top_k = max(self.top_k, 1)
                logits, token_ids = logits.topk(k=self.top_k, sorted=False)

            # Sample from a truncated distribution containing only the most probable tokens whose cumulative probability
            # is no greater than some parameter P (the 'nucleus')
            if self.top_p < 1.0:
                logits, token_ids = logits.sort(descending=True)
                probs = logits.softmax(dim=-1)
                cum_probs = probs.cumsum(dim=-1)

                unlikely_token_mask = (cum_probs > self.top_p)
                unlikely_token_mask[..., :1] = False  # Exclude the very most probable token from being removed
                probs[unlikely_token_mask] = 0.0
            else:
                probs = logits.softmax(dim=-1)

            indices = probs.multinomial(num_samples=1).view(*logits.shape[:-1], 1)
            token_ids = indices if token_ids is None else token_ids.gather(dim=-1, index=indices).squeeze(-1)

        token_ids = token_ids.flatten()  # (batch_size * num_beams)
        self.output_ids[self.live_sample_mask, self.current_index] = token_ids.type_as(self.output_ids)

        self.current_index += 1
        continuing_sample_mask = (token_ids != self.end_token) & (self.current_index < self.output_ids.shape[-1])
        self.live_sample_mask[self.live_sample_mask.clone()] &= continuing_sample_mask

        return continuing_sample_mask

    def should_stop(self) -> bool:
        return self.current_index >= self.output_ids.shape[-1] - 1 or not self.live_sample_mask.any()

    def final_output(self) -> Tensor:
        return self.output_ids[:, 1:]   # Don't include the start token
