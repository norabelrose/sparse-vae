from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from itertools import chain
from math import ceil
from tokenizers import Tokenizer
from torch import Tensor
from typing import Dict, Iterable, List, Tuple, Union
import numpy as np
import random
import torch


# Convert text into WordPiece tokens, while also saving some important stats. This is set up as a freestanding
# function to avoid an annoying crash that happens when dill, a HuggingFace dependency, tries to pickle the function
def tokenize(batch, tokenizer: Tokenizer, chunk: bool, min_tokens: int, max_tokens: int) -> Dict[str, list]:
    raw_encodings = tokenizer.encode_batch(batch['text'])
    if chunk:
        raw_encodings = [[x] + x.overflowing for x in raw_encodings]
        raw_encodings = list(chain.from_iterable(raw_encodings))

    char_counts, token_ids, word_ids = [], [], []
    for encoding, original in zip(raw_encodings, batch['text']):
        ids = encoding.ids
        if min_tokens <= len(ids) <= max_tokens:
            char_counts.append(len(original))
            token_ids.append(ids)

            # The Encoding.word_ids property has a None element for special tokens,
            # but it's easier to work with if we use -1
            word_ids.append([-1 if w is None else w for w in encoding.word_ids])

    return {
        'text': token_ids,
        'num_char': char_counts,
        'num_tokens': [len(x) for x in token_ids],
        'num_words': [max(words) for words in word_ids]
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
class PrebatchedRandomSampler:
    batches: List[Tuple[int, int]]

    def __post_init__(self):
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
        while length <= 0:
            start, length = self.remaining_batches.pop()
            print(f"Warning: Found zero-length batch w/ start index {start}. Skipping...")

        return list(range(start, start + length))


# Get a list of the columns in a Dataset or DatasetDict, asserting that they are the same across splits if the latter
def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> List[str]:
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))

        assert all(cols == columns for cols in cols_by_split), "All splits must have the same columns"
        return columns

    return dataset.column_names


# Get a list of the features in a Dataset or DatasetDict, asserting that they are the same across splits if the latter
def get_features_all_equal(dataset: Union[Dataset, DatasetDict]) -> dict:
    if isinstance(dataset, DatasetDict):
        features_list = [split.features for split in dataset.values()]
        first_features = features_list[0]

        assert all(features == first_features for features in features_list), "All splits must have the same features"
        return first_features

    return dataset.features

def total_dataset_len(dataset: Union[Dataset, DatasetDict]) -> int:
    return sum(len(split) for split in dataset.values()) if isinstance(dataset, DatasetDict) else len(dataset)

def compute_uniform_sized_batches(lengths: Iterable[int], indices: List[int], max_size: int) -> Dict[str, list]:
    # We're assuming the lengths are already sorted
    token_cumcounts = np.cumsum(lengths)
    starts, batch_sizes = [indices[0]], []

    while len(token_cumcounts) > 0:
        batch_size = np.searchsorted(token_cumcounts, token_cumcounts[0] + max_size) - 1
        token_cumcounts = token_cumcounts[batch_size + 1:]
        batch_sizes.append(batch_size); starts.append(starts[-1] + batch_size)

    return {'start': starts[:-1], 'length': batch_sizes}
