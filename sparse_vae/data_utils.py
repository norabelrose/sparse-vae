from dataclasses import dataclass
from datasets import Dataset, DatasetDict
from itertools import chain
from tokenizers import Tokenizer
from typing import Dict, List, Tuple, Union
import numpy as np
import random

from torch.utils.data.sampler import Sampler


# Convert text into WordPiece tokens, while also saving some important stats. This is set up as a freestanding
# function to avoid an annoying crash that happens when dill, a HuggingFace dependency, tries to pickle the function
def tokenize(batch, tokenizer: Tokenizer, chunk: bool) -> Dict[str, list]:
    raw_text = batch['text']
    raw_encodings = tokenizer.encode_batch(raw_text)
    if chunk:
        raw_encodings = [[x] + x.overflowing for x in raw_encodings]
        raw_encodings = list(chain.from_iterable(raw_encodings))

    token_ids = [encoding.ids for encoding in raw_encodings]
    return {
        'text': token_ids,
        'num_bytes': [len(bytes(x, 'utf8')) for x in raw_text],
        'num_tokens': [len(x) for x in token_ids]
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
        assert length > 0, "Something's wrong, found a found zero-length batch"
        return list(range(start, start + length))


@dataclass
class UniformSizeRandomSampler(Sampler):
    documents: List[Tuple[int, int]]  # (document index, length bin) tuples
    max_size: int

    def __post_init__(self):
        assert all(doc_len <= self.max_size for idx, doc_len in self.documents)
        self._compute_batches()

    def _compute_batches(self):
        # Python's list.sort() implementation is guaranteed to be stable, so
        # documents will be shuffled within each length bin
        random.shuffle(self.documents)
        self.documents.sort(key=lambda doc: doc[1])

        # Each batch is a list of document indices
        self.batches = [[]]
        cur_max_doc_len = 0

        for doc_idx, doc_len in self.documents:
            # Would adding this sample to the batch cause us to exceed the max token count?
            # If so, create a new batch and add this sample to it
            cur_max_doc_len = max(cur_max_doc_len, doc_len)
            batch_numel = cur_max_doc_len * (len(self.batches[-1]) + 1)

            if batch_numel > self.max_size:
                cur_max_doc_len = doc_len
                self.batches.append([doc_idx])
            else:
                self.batches[-1].append(doc_idx)

        # Now shuffle the batches so that we visit them in random order
        random.shuffle(self.batches)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batches)

    def __next__(self):
        if not self.batches:
            self._compute_batches()
            raise StopIteration

        batch = self.batches.pop()
        assert batch, "Something's wrong, found a found zero-length batch"
        return batch


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

def compute_uniform_sized_batches(lengths: List[int], max_size: int) -> Dict[str, list]:
    starts = [0]
    cur_token_count = 0

    for i, length in enumerate(lengths):
        assert length <= max_size, f"Found a document with {length} tokens, but the max tokens per batch is {max_size}"

        # Would adding this sample to the batch cause us to exceed the max token count?
        # If so, create a new batch and add this sample to it
        cur_token_count += length
        if cur_token_count > max_size:
            cur_token_count = length
            starts.append(i)

    return {'start': starts, 'length': np.diff(starts, append=len(lengths))}
