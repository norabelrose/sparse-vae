from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from itertools import chain
from tokenizers import Tokenizer
from typing import Dict, Iterable, List, Tuple, Union
import numpy as np
import random


# Convert text into WordPiece tokens, while also saving some important stats. This is set up as a freestanding
# function to avoid an annoying crash that happens when dill, a HuggingFace dependency, tries to pickle the function
def tokenize(batch, tokenizer: Tokenizer, chunk: bool) -> Dict[str, list]:
    raw_text = batch['text']
    raw_encodings = tokenizer.encode_batch(raw_text)
    if chunk:
        raw_encodings = [[x] + x.overflowing for x in raw_encodings]
        raw_encodings = list(chain.from_iterable(raw_encodings))

    char_counts, token_ids, word_ids = [], [], []
    for encoding, original in zip(raw_encodings, raw_text):
        ids = encoding.ids
        # if min_tokens <= len(ids) <= max_tokens:
        char_counts.append(len(original))
        token_ids.append(ids)

        # The Encoding.word_ids property has a None element for special tokens,
        # but it's easier to work with if we use -1
        word_ids.append([-1 if w is None else w for w in encoding.word_ids])

    return {
        'text': token_ids,
        'num_bytes': [len(bytes(x, 'utf8')) for x in raw_text],
        'num_char': char_counts,
        'num_tokens': [len(x) for x in token_ids],
        'num_words': [max(words) for words in word_ids]
    }


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
    last_cumcount = 0

    while len(token_cumcounts) > 0:
        # searchsorted returns the index `i` s.t. cumcounts[i - 1] <= next cumcount < cumcounts[i]; therefore
        # a batch from document 0 up to but not including document `i` will be the largest possible batch that
        # doesn't exceed the max token count, and `i` will be the number of documents in the batch
        batch_size = np.searchsorted(token_cumcounts, last_cumcount + max_size, side='right')
        last_cumcount = token_cumcounts[batch_size - 1]
        token_cumcounts = token_cumcounts[batch_size:]
        batch_sizes.append(batch_size); starts.append(starts[-1] + batch_size)

    return {'start': starts[:-1], 'length': batch_sizes}
