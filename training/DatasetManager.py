from Utilities import *

import codecs
import csv
import random
import requests
import zlib

from pathlib import Path
from tokenizers import BertWordPieceTokenizer, Encoding
from tqdm.auto import tqdm

IO_DEFAULT_CHUNK_SIZE = 5242880  # 5 MB

class DatasetManager:
    datasets_to_urls: ClassVar[dict] = {
        'all_the_news': 'https://www.dropbox.com/s/cn2utnr5ipathhh/all-the-news-2-1.zip'
    }

    # Lazily loaded unless explicitly set by someone else
    tokenizer: BertWordPieceTokenizer = None

    @classmethod
    def _lazily_load_tokenizer(cls):
        if cls.tokenizer:
            return

        vocab_path = Path(__file__).parent.parent / 'resources' / 'pretrained-vocab.txt'
        cls.tokenizer = BertWordPieceTokenizer.from_file(vocab_path, lowercase=True)

    # Iterates over chunks of the bytes of a local file
    @staticmethod
    def _iterate_local_file_bytes(filepath: str, chunk_size: int = IO_DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
        with open(filepath, 'rb') as handle:
            while True:
                chunk = handle.read(chunk_size)
                if chunk:
                    yield chunk
                else:
                    break

    # Basically a convenient wrapper around response.iter_content() from requests, with built-in tqdm
    @staticmethod
    def _iterate_remote_bytes(url: str, chunk_size: int = IO_DEFAULT_CHUNK_SIZE, use_tqdm: bool = True,
                              tqdm_desc: str = 'Downloading') -> Iterator[bytes]:
        response = requests.get(url, stream=True)
        raw_length: Optional[str] = response.headers.get('content-length')
        total_length = int(raw_length) if raw_length else None

        loop_iterator = response.iter_content(chunk_size)
        if use_tqdm:
            loop_iterator = tqdm(loop_iterator, desc=tqdm_desc, total=total_length)

        yield from loop_iterator

    # Iterates over chunks of decompressed bytes given an iterator over compressed (e.g. zipped) bytes; that is,
    # it iteratively unzips a stream of bytes.
    @staticmethod
    def _iterate_decompressed_bytes(byte_iterator: Iterator[bytes]) -> Iterator[bytes]:
        decompressor = zlib.decompressobj(32 + zlib.MAX_WBITS)
        for chunk in byte_iterator:
            data = decompressor.decompress(chunk)

            while len(decompressor.unused_data):
                leftovers = decompressor.unused_data  # end of one block
                decompressor = zlib.decompressobj(32 + zlib.MAX_WBITS)  # create a new decompressor
                data += decompressor.decompress(leftovers)  # decompress the leftovers

            yield data

    # Iterates over bytes given a URL to a zipfile; that is, it unzips and downloads at the same time.
    @classmethod
    def _iterate_remote_zipfile_bytes(cls, url: str, chunk_size: int = IO_DEFAULT_CHUNK_SIZE, use_tqdm: bool = True,
                                      tqdm_desc: str = 'Downloading and unzipping') -> Iterator[bytes]:
        compressed_bytes_iterator = cls._iterate_remote_bytes(url, chunk_size, use_tqdm, tqdm_desc)
        yield from cls._iterate_decompressed_bytes(compressed_bytes_iterator)

    # Iterates text samples from a dataset in CSV or JSON format (as determined by the data_format parameter).
    # sample_key is the name of the CSV column, or the JSON dictionary key, which contains the text we want.
    @staticmethod
    def _iterate_text_samples_from_dataset(data_iterator: Iterator[bytes], data_format: str,
                                           sample_key: str = 'text') -> Iterator[str]:
        # We assume the dataset is in text form (CSV or JSON). Could be from a remote or local file.
        unicode_iterator = codecs.iterdecode(data_iterator, encoding='utf-8', errors='ignore')
        if data_format == 'csv':
            csv_iterator = csv.reader(unicode_iterator)
            column_names: List[str] = next(csv_iterator)  # First row contains the column names

            try:
                text_column = column_names.index(sample_key)
            except ValueError:
                raise ValueError(f"CSV dataset does not contain a column named '{sample_key}'. "
                                 f"Found columns: {column_names}")

            for row in csv_iterator:
                yield row[text_column]

        elif data_format == 'json':
            raise NotImplementedError
        else:
            raise NotImplementedError

    @classmethod
    def _iterate_tokenized_text_samples(cls, sample_iterator: Iterator[str], batch_size: int = 1000,
                                        shuffle: bool = True) -> Iterator[List[Encoding]]:
        cls._lazily_load_tokenizer()

        # Create batches of strings so that we can use tokenizer.encode_batch, which has better performance
        # (due to parallelism) according to Narsil here: https://github.com/huggingface/tokenizers/issues/398.
        # This way it's also easy to batch our file I/O when we write this to a binary file.
        while True:
            sample_batch = []
            should_break_flag = False

            for _ in range(batch_size):
                try:
                    next_sample = next(sample_iterator)
                except StopIteration:
                    should_break_flag = True    # We ran out of data
                else:
                    sample_batch.append(next_sample)

            # Shuffle the order of the individual text samples (within this batch)
            if shuffle:
                random.shuffle(sample_batch)

            yield cls.tokenizer.encode_batch(sample_batch)

            if should_break_flag:
                break

    # Downloads, unzips, tokenizes, and dumps to a binary file all at once
    @classmethod
    def _generate_tokenized_file_from_url(cls, filepath: str, url: str, data_format: str, sample_key: str,
                                          use_tqdm: bool = True,
                                          tqdm_desc: str = 'Downloading, unzipping, tokenizing, and saving'):
        bytes_iterator = cls._iterate_remote_zipfile_bytes(url, use_tqdm=use_tqdm, tqdm_desc=tqdm_desc)
        text_sample_iterator = cls._iterate_text_samples_from_dataset(bytes_iterator, data_format, sample_key)
        tokenized_sample_iterator = cls._iterate_tokenized_text_samples(text_sample_iterator)

        with open(filepath, 'wb') as handle:
            for encodings in tokenized_sample_iterator:
                flattened_ints = [x for x in encoding.ids for encoding in encodings]
                handle.write(bytes(flattened_ints))
