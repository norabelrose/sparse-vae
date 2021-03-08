from typing import *
from dataclasses import dataclass
from math import ceil
from tokenizers import Tokenizer
from torch import Tensor
import random
import torch


# Flattens large lists faster than list(itertools.chain.from_iterable(x))
def fast_flatten(original: List[List]) -> List:
    # Pre-allocate memory
    total_size = sum(len(x) for x in original)
    output = [None] * total_size

    cur_idx = 0
    for x in original:
        next_idx = cur_idx + len(x)
        output[cur_idx:next_idx] = x
        cur_idx = next_idx

    return output


# Convert text into WordPiece tokens, while also saving some important stats. This is set up as a freestanding
# function to avoid an annoying crash that happens when dill, a HuggingFace dependency, tries to pickle the function
def tokenize(batch: Dict[str, list], tokenizer: Tokenizer, should_chunk: bool, min_tokens: int,
              max_tokens: int) -> Dict[str, list]:
    if should_chunk:
        # Tokenizer has had .enable_truncation(max_tokens) called on it
        encodings = tokenizer.encode_batch(batch['text'])
        encodings = [[x] + x.overflowing for x in encodings]
        for sample in encodings:
            if len(sample[-1].ids) < min_tokens:  # Only the last sequence might possibly be too short
                encodings.pop()

        encodings = fast_flatten(encodings)
    else:
        encodings = tokenizer.encode_batch(batch['text'])
        encodings = [x for x in encodings if min_tokens <= len(x.ids) <= max_tokens]

    token_batches = [x.ids for x in encodings]

    # The Encoding.word_ids property has a None element for special tokens, but it's easier to work with if we use -1
    word_batches = [[-1 if w is None else w for w in x.word_ids]
                    for x in encodings]

    return {
        'text': token_batches,
        'word_ids': word_batches,
        'num_tokens': [len(x) for x in token_batches],
        'num_words': [max(word_ids) for word_ids in word_batches]
    }


# Performs whole-word masking on the LongTensor `tokens` in-place, and returns a BoolTensor indicating the positions
# that were masked out.
def whole_word_mask_(tokens: Tensor, word_counts: Tensor, word_ids: Tensor, mask_prob: float, mask_token: int) -> Tensor:
    one = torch.tensor([1.0])
    logits = [one.expand(count if count > 0 else 1) for count in word_counts]
    logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True)

    mask_counts = word_counts.mul(mask_prob).round().long()
    masked_words = logits.multinomial(num_samples=mask_counts.max()).unsqueeze(-2)

    noise_mask = word_ids.unsqueeze(-1).eq(masked_words).any(dim=-1)
    tokens[noise_mask] = mask_token

    return noise_mask

# Used to return random batches that are of roughly uniform length
@dataclass
class ContiguousRandomSampler:
    dataset_length: int
    batch_size: int

    def __post_init__(self):
        self.batch_starts = list(range(0, self.dataset_length, self.batch_size))

    def __iter__(self):
        return self

    def __len__(self):
        return int(ceil(self.dataset_length / self.batch_size))

    def __next__(self):
        if not self.batch_starts:
            self.batch_starts = list(range(0, self.dataset_length, self.batch_size))  # Prepare for next epoch
            raise StopIteration

        index_idx = random.randrange(0, len(self.batch_starts))
        start_idx = self.batch_starts[index_idx]
        end = start_idx + self.batch_size
        if end > self.dataset_length:
            end = self.dataset_length

        del self.batch_starts[index_idx]  # Don't yield the same indices twice
        return list(range(start_idx, end))
