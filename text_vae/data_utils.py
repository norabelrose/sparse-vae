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
        random.shuffle(self.batch_starts)

    def __iter__(self):
        return self

    def __len__(self):
        return int(ceil(self.dataset_length / self.batch_size))

    def __next__(self):
        if not self.batch_starts:
            self.batch_starts = list(range(0, self.dataset_length, self.batch_size))  # Prepare for next epoch
            random.shuffle(self.batch_starts)
            raise StopIteration

        start_idx = self.batch_starts.pop()
        end = start_idx + self.batch_size
        if end > self.dataset_length:
            end = self.dataset_length

        return list(range(start_idx, end))

@dataclass
class UniformSizeRandomSampler:
    sample_lengths: List[int]
    max_tokens_per_batch: int

    def __post_init__(self):
        self.batches = compute_uniform_sized_batches(self.sample_lengths, self.max_tokens_per_batch)
        self.remaining_batches = self.batches.copy()
        random.shuffle(self.remaining_batches)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batches)

    def __next__(self):
        if not self.remaining_batches:
            self.remaining_batches = self.batches.copy()
            random.shuffle(self.remaining_batches)
            raise StopIteration

        start, length = self.remaining_batches.pop()
        return list(range(start, start + length))


def compute_uniform_sized_batches(lengths: List[int], max_size: int) -> List[Tuple[int, int]]:
    cur_num_tokens = 0  # Running total of the number of tokens in this batch
    cur_num_samples = 0  # Length in *number of samples*
    cur_start = 0  # Index of the first *sample* in this batch
    batches = []  # List of tuples of the form (first sample index, num samples)

    for i, num_tokens in enumerate(lengths):
        # If adding this sample to the current batch would make it too big, end the current batch
        # and add the sample to a new batch
        if cur_num_tokens + num_tokens > max_size:
            if cur_num_samples > 0:
                batches.append((cur_start, cur_num_samples))

            # If this sample is so huge that it exceeds the maximum number of tokens by itself,
            # then just skip it and see if the next sample is of a more manageable size
            if num_tokens > max_size:
                cur_start = i + 1
                cur_num_samples = 0
                cur_num_tokens = 0
            else:
                cur_start = i
                cur_num_samples = 1
                cur_num_tokens = num_tokens
        else:
            cur_num_tokens += num_tokens
            cur_num_samples += 1

    # Take care of whatever is left over when we're done iterating over all the samples
    if cur_num_samples > 0:
        batches.append((cur_start, cur_num_samples))

    return batches

# Actually chunk large batches of samples from a HuggingFace dataset into uniformly sized minibatches.
# Designed to be used with the dataset.map() method.
def perform_uniform_size_batching(batch: Dict[str, list], max_size: int, length_key: str) -> Dict[str, list]:
    batch_tuples = compute_uniform_sized_batches(batch[length_key], max_size)
    return {
        key: [value[start:start + length] for start, length in batch_tuples]
        for key, value in batch.items()
    }
