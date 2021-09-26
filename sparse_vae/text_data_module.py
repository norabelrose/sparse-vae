from datasets import Features, Value, Sequence, concatenate_datasets, load_dataset
from multiprocessing import cpu_count
from omegaconf import DictConfig
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch import Tensor
from typing import Any, Optional, cast
from .core import PaddedTensor
from .data_utils import *
import logging
import numpy as np
import os
import pytorch_lightning as pl
import torch


@dataclass
class TextDataModuleHparams:
    tokens_per_batch: Optional[int] = 50_000

    chunk_documents: bool = False
    dataset_name: str = 'wikipedia'
    dataset_config: Optional[str] = '20200501.en'
    dataset_path: Optional[str] = None
    min_tokens_per_sample: int = 512
    max_tokens_per_sample: int = 25_000
    split: Optional[str] = None         # Any string of the format supported by the HuggingFace datasets library
    vocab_size: int = 2 ** 15


# Base class for Text VAE data modules- takes care of boilerplate
# noinspection PyAbstractClass
class TextDataModule(pl.LightningDataModule):
    def __init__(self, hparams: DictConfig):
        super(TextDataModule, self).__init__()

        # Silence the very annoying wall of "Loading cached processed dataset" messages from HuggingFace datasets
        handler = logging.StreamHandler()
        handler.addFilter(lambda log_record: not log_record.getMessage().startswith('Loading cached'))
        logging.getLogger('datasets').addHandler(handler)

        # Avoid running out of shared memory for IPC between dataloader workers and the main process
        # torch.multiprocessing.set_sharing_strategy('file_system')
        # self.dataset: Optional[Union[Dataset, DatasetDict]] = None
        self.extra_start_tokens = 0
        self.hparams = hparams

        # This needs to be a multiple of the sparse attention block size (usually set to 16). Larger
        # padding multiples have the benefit of requiring fewer lookup tables to be computed, load-
        # balanced, and cached at the beginning of training.
        self.pad_to_multiple_of = 512

        self._preproc_batch_size = None
        self._tokenizer = None
        self.bytes_per_token = torch.ones(hparams.vocab_size)

        # Make sure dataset save dir exists
        (Path.cwd() / 'sparse-vae-datasets').mkdir(exist_ok=True)

    # Finds a reasonable (per thread) batch size given the average length of the samples
    @property
    def preproc_batch_size(self):
        if not self._preproc_batch_size:
            prng = np.random.default_rng(seed=7295)  # Make this deterministic

            # If we have a dataset with multiple splits, use the largest split
            if not isinstance(self.dataset, DatasetDict):
                dataset = self.dataset
            else:
                dataset = max(self.dataset.values(), key=lambda x: x.num_rows)

            # Get Monte Carlo estimate of the average sample length without actually iterating through the whole dataset
            indices = prng.choice(len(dataset), 10, replace=False)
            elements = [cast(dict, dataset[int(i)]) for i in indices]
            avg_len_estimate = sum(len(x['text']) for x in elements) / len(elements)

            # 1,000 for 1,000 character samples
            self._preproc_batch_size = round(1e6 / avg_len_estimate)

        return self._preproc_batch_size

    @property
    def tokenizer(self) -> ByteLevelBPETokenizer:
        if not self._tokenizer:
            self.setup_tokenizer()

        return self._tokenizer

    def create_dataset(self):
        if path := self.hparams.dataset_path:
            self.dataset = DatasetDict.load_from_disk(path)
            self.hparams.dataset_name = self.dataset['train'].config_name
        else:
            self.dataset = cast(DatasetDict, load_dataset(
                self.hparams.dataset_name, name=self.hparams.dataset_config,
                split=self.hparams.split, cache_dir=str(Path.cwd() / 'sparse-vae-datasets')
            ))

    def prepare_data(self, *args, **kwargs):
        self.create_dataset()   # Download or load a big file from disk

        features = get_features_all_equal(self.dataset)
        assert (text_column := features.get('text')), "Can't find text colum in dataset"

        # Dataset is already pre-tokenized
        if hasattr(text_column, 'feature') and text_column.feature.dtype.startswith('int'):
            print(f"Computing sample lengths...")
            self.dataset = self.dataset.map(
                lambda batch: {'num_tokens': [len(row) for row in batch['text']]}, batched=True, batch_size=1000
            )

        # We need to tokenize the raw text
        else:
            # We do this to make sure that the token IDs aren't stored as 64 bit integers,
            # which can increase the disk space used by 4x
            feats = features.keys()
            feat_dtypes = {
                'num_bytes': Value('int32'),
                'num_tokens': Value('int32'),
                'text': Sequence(Value('uint16')),
                'title': Value('string')
            }

            # For Project Gutenberg 1919, the original train/val/test split is
            # is terrible- val and test splits are too small- so we redo the split
            if self.hparams.dataset_name == 'pg19':
                self.dataset = concatenate_datasets(list(self.dataset.values()))
                self.dataset = self.dataset.rename_column('short_book_title', 'title')  # type: ignore
                self.dataset = self.dataset.remove_columns(['publication_date', 'url'])

            # For PG19 & Wikipedia articles
            if 'title' in feats:
                feat_dtypes['title'] = Value('string')

            # For Yelp
            if 'label' in feats:
                feat_dtypes['label'] = Value('uint8')

            print(f"Tokenizing '{self.hparams.dataset_name}'...")
            self.dataset = self.dataset.map(
                tokenize, batched=True, batch_size=1000, features=Features(feat_dtypes),
                fn_kwargs=dict(chunk=self.hparams.chunk_documents, tokenizer=self.tokenizer)
            )

        total_docs = len(self.dataset)
        min_tokens = self.hparams.min_tokens_per_sample
        max_tokens = self.hparams.max_tokens_per_sample
        self.dataset = self.dataset.filter(
            lambda n: min_tokens <= n <= max_tokens, input_columns='num_tokens', num_proc=min(10, cpu_count()),
        )
        print(f"Training on {len(self.dataset)} of {total_docs} documents.")

        # Silence annoying warnings from the tokenizers package after we spawn DataLoader processes
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Ensure that there actually is a train-test split
        if not isinstance(self.dataset, DatasetDict):
            test_size = min(50_000, round(len(self.dataset) * 0.05))
            self.dataset = self.dataset.train_test_split(test_size=test_size, shuffle=True)  # noqa

        # Annoyingly the wikipedia dataset *is* a DatasetDict but it only has a 'train' split
        elif 'test' not in self.dataset:
            self.dataset = self.dataset['train'].train_test_split(test_size=50_000, shuffle=True)  # noqa

        # Round up all the token counts to the nearest block size, to account for padding
        with self.dataset.formatted_as('numpy'):
            self.dataset = self.dataset.map(
                lambda token_count, bin_size: {'length_bin': token_count + (bin_size - token_count % bin_size)},
                batched=True, input_columns=['num_tokens'], fn_kwargs=dict(bin_size=self.pad_to_multiple_of)
            )

    def setup(self, stage: Optional[str] = None):
        self.dataset.set_format('numpy')    # We manually call torch.from_numpy to handle uint16 conversion

    def train_dataloader(self, split: str = 'train') -> DataLoader:
        data = cast(TorchDataset[Dict[str, Any]], self.dataset[split])
        return DataLoader(
            data, batch_sampler=UniformSizeRandomSampler(
                documents=list(enumerate(data['length_bin'])),
                max_size=self.hparams.tokens_per_batch
            ),
            collate_fn=self.collate, num_workers=10, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return self.train_dataloader(split='test')

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()

    def predict_dataloader(self) -> List[DataLoader]:
        return [self.train_dataloader(), self.val_dataloader()]

    def collate(self, inputs: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        # Annoyingly PyTorch doesn't have a uint16 type- check if the vocab size would overflow an
        # int16; if so we copy into an int32 tensor, otherwise we do a no-copy reinterpret cast to int16
        upcast = self.tokenizer.get_vocab_size() > 2 ** 15
        batch = {
            'num_bytes': torch.tensor([x['num_bytes'] for x in inputs]),
            'num_tokens': torch.tensor([x['num_tokens'] for x in inputs]),
            'token_ids': PaddedTensor.from_raw(self.pad_pack([
                torch.from_numpy(x['text'].view(np.int16) if not upcast else x['text'].astype(np.int32))
                for x in inputs
            ]))
        }
        # Yelp reviews
        if 'label' in inputs[0]:
            batch['label'] = torch.tensor([x['label'] for x in inputs])

        return batch

    def pad_pack(self, batch: List[Tensor], pad_value: int = 0) -> Tensor:
        extras = self.extra_start_tokens
        buffer_len = max(len(x) for x in batch) + extras

        factor = self.pad_to_multiple_of
        if factor > 1:
            remainder = buffer_len % factor
            buffer_len += factor - remainder if remainder else 0

        buffer = torch.full([len(batch), buffer_len], pad_value, dtype=batch[0].dtype)
        if extras:
            buffer[:, :extras] = self.start_token

        for i, sequence in enumerate(batch):
            buffer[i, extras:len(sequence) + extras] = sequence

        return buffer

    # Called from prepare_data, as well as from setup when we load from a checkpoint
    def setup_tokenizer(self):
        tokenizers_dir = Path.cwd() / 'sparse-vae-pretrained' / 'tokenizers'
        tokenizers_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = tokenizers_dir / (self.hparams.dataset_name + '.json')

        if vocab_path.exists():
            print(f'Loading pretrained tokenizer from {vocab_path}')
            self._tokenizer = Tokenizer.from_file(str(vocab_path))
            assert self.tokenizer.get_vocab_size() == self.hparams.vocab_size
        else:
            print(f'Training a BPE tokenizer for the dataset {self.hparams.dataset_name}')
            self._tokenizer = ByteLevelBPETokenizer()
            self._tokenizer.post_processor = RobertaProcessing(sep=("[SEP]", 2), cls=("[CLS]", 1))
            batch_size = self.preproc_batch_size

            def text_iterator(dataset):
                data_len = len(dataset)
                for i in range(0, data_len, batch_size):
                    end = i + batch_size
                    if end > data_len:
                        end = data_len

                    yield dataset[i:end]['text']

            # Don't train the vocabulary on the validation dataset just to be safe
            data = self.dataset['train'] if isinstance(self.dataset, DatasetDict) else self.dataset
            self.tokenizer.train_from_iterator(
                text_iterator(data),
                vocab_size=self.hparams.vocab_size,
                special_tokens=["[PAD]", "[CLS]", "[SEP]"]
            )
            self.tokenizer.save(str(vocab_path))

        # Store the number of bytes per BPE token so we can easily compute and log the bits-per-byte metric
        for token, token_id in self._tokenizer.get_vocab().items():
            # Special tokens ([PAD], [CLS], [SEP]) are encoded with IDs 0, 1, and 2 and are considered to be 1 byte
            self.bytes_per_token[token_id] = len(token.encode()) if token_id > 2 else 1

        # This hyperparameter could change every run of the application so we don't want to save
        # the truncation policy with the tokenizer
        self.start_token = self.tokenizer.get_vocab()['[CLS]']
        if self.hparams.chunk_documents:
            self._tokenizer.enable_truncation(self.hparams.max_tokens_per_sample)
