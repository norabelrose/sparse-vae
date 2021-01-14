from argparse import ArgumentParser, Namespace
from itertools import chain, islice
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.utils.data import DataLoader
from typing import *
import datasets
import multiprocessing
import pytorch_lightning as pl
import torch
import warnings


# Base class for Text VAE data modules- takes care of boilerplate
# noinspection PyAbstractClass
class TextVaeDataModule(pl.LightningDataModule):
    dataset_name: ClassVar[str] = 'dataset'  # Should be overridden by subclasses

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = super(TextVaeDataModule, cls).add_argparse_args(parent_parser)
        parent_parser.add_argument("--split", type=str, default='train')
        return parent_parser

    @classmethod
    def from_argparse_args(cls, args: Namespace, **kwargs):
        new_module = super(TextVaeDataModule, cls).from_argparse_args(args)
        return new_module

    def __init__(self, batch_size: int = 10, max_sample_length: int = 512, chunk_long_samples: bool = True,
                 dataset_save_dir: Optional[Path] = None):
        super(TextVaeDataModule, self).__init__()

        # These warnings are spurious and seem to pop up due to a bug in PyTorch which was fixed in PR #47160
        warnings.filterwarnings('ignore', message='The given NumPy array is not writeable')

        self.batch_size = batch_size
        self.chunk_long_samples = chunk_long_samples
        self.max_sample_length = max_sample_length
        self.dataset = None     # HuggingFace Dataset object, possibly with both train and test splits

        vocab_path = Path(__file__).parent / 'resources' / 'pretrained-vocab.txt'
        self.tokenizer = BertWordPieceTokenizer.from_file(str(vocab_path), lowercase=True)

        # Get path to store the processed dataset
        self.dataset_dir = dataset_save_dir or Path.cwd() / 'text-vae-datasets'
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # Subclass hook
    def create_dataset(self):
        self.dataset = datasets.load_dataset(self.dataset_name, split='train')

    def prepare_data(self, *args, **kwargs):
        # Check if we already have the dataset
        processed_path = self.dataset_dir / self.dataset_name
        if processed_path.exists():
            self.dataset = datasets.load_from_disk(str(processed_path))
            return

        self.create_dataset()   # Download or load a big file from disk
        assert self.dataset

        sep_token: int = self.tokenizer.get_vocab()['[SEP]']
        def tokenize_and_chunk(batch: Dict[str, list]) -> Dict[str, list]:
            encodings = self.tokenizer.encode_batch(batch['text'])                   # Tokenize
            id_iter = chain.from_iterable([encoding.ids for encoding in encodings])  # Chain all samples together

            # Break into chunks, and tack on a [SEP] token to the end of each one
            sep_delimited_iter = chain(islice(id_iter, self.max_sample_length - 1), [sep_token])
            chunk_iter = iter(lambda: list(sep_delimited_iter), [])

            return {'text': list(chunk_iter)}

        nontext_cols = self.dataset.column_names
        del nontext_cols['text']

        # Tokenize, chunk, and save
        size = max(multiprocessing.cpu_count(), 10)
        print(f"Tokenizing and chunking '{self.dataset_name}'...")
        self.dataset = self.dataset.map(tokenize_and_chunk, batched=True, batch_size=size, remove_columns=nontext_cols)
        self.dataset.rename_column_('text', 'token_ids')

        print(f"Saving '{self.dataset_name}'...")
        self.dataset.save_to_disk(self.dataset_dir / self.dataset_name)

    def setup(self, stage: Optional[str] = None):
        self.dataset = self.dataset.train_test_split(test_size=0.05, shuffle=True)
        self.dataset.set_format('torch')

    @staticmethod
    def collate(inputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        # Combine into a single batched and padded tensor
        inputs = torch.nn.utils.rnn.pad_sequence([x['token_ids'] for x in inputs], batch_first=True)
        return {'token_ids': inputs, 'padding_mask': inputs.eq(0)}

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True, collate_fn=self.collate,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset['test'], batch_size=self.batch_size, collate_fn=self.collate,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()


# noinspection PyAbstractClass
class ProjectGutenbergDataModule(TextVaeDataModule):
    dataset_name: ClassVar[str] = 'pg19'
