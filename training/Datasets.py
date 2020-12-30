from itertools import chain, islice
from pathlib import Path
from pyarrow.csv import ConvertOptions
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Optional, Union
import datasets
import os
import pytorch_lightning as pl
import torch


# Base class for Text VAE data modules- takes care of boilerplate
class TextVaeDataModule(pl.LightningDataModule):
    dataset_name = 'dataset'  # Should be overridden by subclasses

    def __init__(self, batch_size: int = 10, max_sample_length: int = 512):
        super().__init__()

        self.batch_size = batch_size
        self.max_sample_length = max_sample_length
        self.dataset = None     # HuggingFace Dataset object with both train and test splits

        # Get path to store the processed dataset
        cache_dir = Path(os.getenv('XDG_CACHE_HOME', '~/.cache'))
        self.dataset_dir = cache_dir.expanduser() / 'text_vae' / 'datasets'
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    # Subclass hook
    def create_dataset(self):
        pass

    def prepare_data(self, *args, **kwargs):
        # Check if we already have the dataset
        processed_path = self.dataset_dir / self.dataset_name
        if processed_path.exists():
            self.dataset = datasets.load_from_disk(processed_path)
            return

        self.create_dataset()   # Download or load a big file from disk
        assert self.dataset

        vocab_path = Path(__file__).parent.parent / 'resources' / 'pretrained-vocab.txt'
        tokenizer = BertWordPieceTokenizer.from_file(vocab_path, lowercase=True)
        tokenizer.enable_padding()
        tokenizer.enable_truncation(max_length=self.max_sample_length)

        def tokenize_batch(batch: List[dict]) -> List[dict]:
            text_samples = [sample['text'] for sample in batch]
            encodings = tokenizer.encode_batch(text_samples)

            return [{'text': encoding.ids} for encoding in encodings]

        def chunk_batch(batch: List[dict]) -> List[dict]:
            id_iterator = chain.from_iterable(sample['text'] for sample in batch)  # Chain all the samples together
            chunks = list(iter(lambda: list(islice(id_iterator, self.max_sample_length)), []))  # Break into chunks

            return [{'text': chunk for chunk in chunks}]

        # Tokenize, chunk, shuffle, split, and save
        self.dataset = self.dataset.map(tokenize_batch, batched=True)
        self.dataset = self.dataset.map(chunk_batch, batched=True)
        self.dataset = self.dataset.train_test_split(test_size=0.05, shuffle=True)
        self.dataset.set_format('torch')
        self.dataset.save_to_disk(self.dataset_dir / self.dataset_name)

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.dataset['test'], batch_size=self.batch_size)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.val_dataloader()

    def transfer_batch_to_device(self, batch: Tensor, device: torch.device) -> Tensor:
        return batch.to(device)


class FunnelPreTrainingDataModule(TextVaeDataModule):
    dataset_name = 'funnel_pretraining'

    def create_dataset(self):
        wikipedia = datasets.load_dataset('wikipedia', '20200501.en', split='train')
        bookcorpus = datasets.load_dataset('bookcorpusopen', split='train')
        openwebtext = datasets.load_dataset('openwebtext', split='train')
        self.dataset = datasets.concatenate_datasets([wikipedia, bookcorpus, openwebtext])


class AllTheNewsDataModule(TextVaeDataModule):
    dataset_name = 'all-the-news'

    def create_dataset(self):
        manager = datasets.DownloadManager('all-the-news', download_config=datasets.DownloadConfig())
        folder = manager.download_and_extract('https://www.dropbox.com/s/cn2utnr5ipathhh/all-the-news-2-1.zip?dl=1')

        self.dataset = datasets.load_dataset('csv', data_files=[os.path.join(folder, 'all-the-news-2-1.csv')],
                                             convert_options=ConvertOptions(include_columns=['article']))
        self.dataset.rename_column_('article', 'text')
