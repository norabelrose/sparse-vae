from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from pyarrow.csv import ConvertOptions
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.utils.data import DataLoader
from typing import ClassVar, Dict, List, Optional, Union
import datasets
import multiprocessing
import os
import pytorch_lightning as pl
import torch


# Base class for Text VAE data modules- takes care of boilerplate
@dataclass
class TextVaeDataModule(pl.LightningDataModule):
    dataset_name: ClassVar[str] = 'dataset'  # Should be overridden by subclasses

    batch_size: int = 10
    max_sample_length: int = 512
    chunk_long_samples: bool = True

    def __post_init__(self):
        self.dataset = None     # HuggingFace Dataset object, possibly with both train and test splits

        vocab_path = os.path.join(os.path.dirname(__file__), 'resources', 'pretrained-vocab.txt')
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_path, lowercase=True)

        # Get path to store the processed dataset
        cache_dir = Path(os.getenv('XDG_CACHE_HOME', '~/.cache'))
        self.dataset_dir = cache_dir.expanduser() / 'text_vae' / 'datasets'
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # Subclass hook
    def create_dataset(self):
        self.dataset = datasets.load_dataset(self.dataset_name)

    def prepare_data(self, *args, **kwargs):
        # Check if we already have the dataset
        processed_path = self.dataset_dir / self.dataset_name
        if processed_path.exists():
            self.dataset = datasets.load_from_disk(processed_path)
            return

        self.create_dataset()   # Download or load a big file from disk
        assert self.dataset

        def tokenize_and_chunk(batch: Dict[str, list]) -> Dict[str, list]:
            encodings = self.tokenizer.encode_batch(batch['text'])                       # Tokenize
            id_iterator = chain.from_iterable([encoding.ids for encoding in encodings])  # Chain all samples together
            chunk_iterator = iter(lambda: list(islice(id_iterator, self.max_sample_length)), [])  # Break into chunks

            return {'text': list(chunk_iterator)}

        nontext_cols = self.dataset.column_names
        nontext_cols.remove('text')

        # Tokenize, chunk, and save
        size = max(multiprocessing.cpu_count(), 10)
        print(f"Tokenizing and chunking '{self.dataset_name}'...")
        self.dataset = self.dataset.map(tokenize_and_chunk, batched=True, batch_size=size, remove_columns=nontext_cols)
        print(f"Saving '{self.dataset_name}'...")
        self.dataset.save_to_disk(self.dataset_dir / self.dataset_name)

    def setup(self, stage: Optional[str] = None):
        self.dataset = self.dataset.train_test_split(test_size=0.05, shuffle=True)
        self.dataset.set_format('torch')

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset['test'], batch_size=self.batch_size)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()

    def transfer_batch_to_device(self, batch: Tensor, device: torch.device) -> Tensor:
        return batch.to(device)


class ProjectGutenbergDataModule(TextVaeDataModule):
    dataset_name: ClassVar[str] = 'pg19'


class AllTheNewsDataModule(TextVaeDataModule):
    dataset_name: ClassVar[str] = 'all-the-news'

    def create_dataset(self):
        manager = datasets.DownloadManager('all-the-news', download_config=datasets.DownloadConfig())
        folder = manager.download_and_extract('https://www.dropbox.com/s/cn2utnr5ipathhh/all-the-news-2-1.zip?dl=1')

        self.dataset = datasets.load_dataset('csv', data_files=[os.path.join(folder, 'all-the-news-2-1.csv')],
                                             convert_options=ConvertOptions(include_columns=['article']))
        self.dataset.rename_column_('article', 'text')
