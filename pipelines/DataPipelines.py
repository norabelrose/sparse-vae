from dataclasses import *
from Pipelines import *

import codecs
import csv
import logging
import random
import requests
import zlib

from tokenizers import BertWordPieceTokenizer, Encoding
from tqdm.auto import tqdm

IO_DEFAULT_CHUNK_SIZE = 5242880  # 5 MB

# Iterates over chunks of the bytes of a local file
@dataclass
class FileStream(Stream[bytes]):
    filepath: str
    chunk_size: int = IO_DEFAULT_CHUNK_SIZE

    def generate(self) -> Iterator[Output]:
        with open(self.filepath, 'rb') as handle:
            if self.iter_counter > 0:
                handle.seek(self.iter_counter * self.chunk_size)  # Seek to where we left off from

            while True:
                chunk = handle.read(self.chunk_size)
                if chunk:
                    yield chunk
                else:
                    break

# Iterates over tensors from a binary file
@dataclass
class TensorFileStream(FileStream):
    pass

# Basically a convenient wrapper around response.iter_content() from requests, with built-in tqdm
@dataclass
class RemoteFileStream(Stream[bytes]):
    url: str
    chunk_size: int = IO_DEFAULT_CHUNK_SIZE
    use_tqdm: bool = True
    tqdm_desc: str = 'Downloading'

    def generate(self) -> Iterator[Output]:
        # Try to seek to the point in the file where we left off last time if applicable
        headers = {"Range": f"bytes={self.iter_counter * self.chunk_size}-"} if self.iter_counter > 0 else None

        response = requests.get(self.url, headers=headers, stream=True)
        loop_iterator = response.iter_content(self.chunk_size)

        if self.iter_counter > 0 and response.status_code != 206:
            logger = logging.getLogger()
            logger.warning(f"Failed to resume download from URL {self.url}; server does not support HTTP Range "
                           f"option. Blocking until we get back to the position we were at before.")

            for _ in range(self.iter_counter):
                next(loop_iterator)

            logger.info("Got back to where we were at.")

        raw_length: Optional[str] = response.headers.get('content-length')
        total_length = int(raw_length) if raw_length else None
        if total_length is not None:
            self.__length_hint__ = lambda: (total_length // self.chunk_size) - self.iter_counter

        if self.use_tqdm:
            loop_iterator = tqdm(loop_iterator, desc=self.tqdm_desc, total=total_length)

        yield from loop_iterator

# Iterates over chunks of decompressed bytes given an iterator over compressed (e.g. zipped) bytes; that is,
# it iteratively unzips a stream of bytes.
@dataclass
class DecompressionFilter(Filter[bytes, bytes]):
    def generate(self) -> Iterator[Output]:
        decompressor = zlib.decompressobj(32 + zlib.MAX_WBITS)
        for chunk in self.inputs:
            data = decompressor.decompress(chunk)

            while len(decompressor.unused_data):
                leftovers = decompressor.unused_data  # end of one block
                decompressor = zlib.decompressobj(32 + zlib.MAX_WBITS)  # create a new decompressor
                data += decompressor.decompress(leftovers)  # decompress the leftovers

            yield data

# Iterates text samples from a dataset in CSV or JSON format (as determined by the data_format parameter).
# sample_key is the name of the CSV column, or the JSON dictionary key, which contains the text we want.
@dataclass
class TextExtractionFilter(Filter[bytes, str]):
    data_format: str
    sample_key: str = 'text'

    def generate(self) -> Iterator[Output]:
        # We assume the dataset is in text form (CSV or JSON). Could be from a remote or local file.
        unicode_iterator = codecs.iterdecode(self.inputs, encoding='utf-8', errors='ignore')
        if self.data_format == 'csv':
            csv_iterator = csv.reader(unicode_iterator)
            column_names: List[str] = next(csv_iterator)  # First row contains the column names
            try:
                text_column = column_names.index(self.sample_key)
            except ValueError:
                raise ValueError(f"CSV dataset does not contain a column named '{self.sample_key}'. "
                                 f"Found columns: {column_names}")
            for row in csv_iterator:
                yield row[text_column]
        elif self.data_format == 'json':
            raise NotImplementedError
        else:
            raise NotImplementedError

@dataclass
class TokenizationFilter(Filter[str, Union[Encoding, bytes]]):
    tokenizer: BertWordPieceTokenizer

    batch_size: int = 1000
    shuffle: bool = True
    return_bytes: bool = False

    def generate(self) -> Iterator[Output]:
        # Create batches of strings so that we can use tokenizer.encode_batch, which has better performance
        # (due to parallelism) according to Narsil here: https://github.com/huggingface/tokenizers/issues/398.
        # This way it's also easy to batch our file I/O when we write this to a binary file.
        while True:
            sample_batch = []
            out_of_data = False
            for _ in range(self.batch_size):
                try:
                    next_sample = next(self.inputs)
                except StopIteration:
                    out_of_data = True
                    break
                else:
                    sample_batch.append(next_sample)

            # Shuffle the order of the individual text samples (within this batch)
            if self.shuffle:
                random.shuffle(sample_batch)

            tokenized_batch = self.tokenizer.encode_batch(sample_batch)

            if self.return_bytes:
                flattened_ints = [x for x in encoding.ids for encoding in tokenized_batch]
                yield bytes(flattened_ints)
            else:
                yield tokenized_batch

            if out_of_data:
                break

@dataclass
class BytesToFile(Target[bytes]):
    filepath: str

    def read(self) -> None:
        with open(self.filepath, 'ab') as handle:   # Note that we use append mode to support resuming
            for chunk in self.inputs:
                handle.write(chunk)
