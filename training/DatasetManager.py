from ..pipelines.DataPipelines import *

from pathlib import Path
from tokenizers import BertWordPieceTokenizer, Encoding

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

    # Downloads, unzips, tokenizes, and dumps to a binary file all at once
    @classmethod
    def _generate_tokenized_file_from_url(cls, filepath: str, url: str, data_format: str, sample_key: str,
                                          use_tqdm: bool = True,
                                          tqdm_desc: str = 'Downloading, unzipping, tokenizing, and saving'):
        cls._lazily_load_tokenizer()

        component_list = [
            RemoteFileStream(url=url, use_tqdm=use_tqdm, tqdm_desc=tqdm_desc),
            TextExtractionFilter(data_format, sample_key),
            TokenizationFilter(tokenizer=cls.tokenizer, return_bytes=True),
            BytesToFile(filepath)
        ]
        if Path(url).suffix in ('.gz', '.zip'):
            component_list.insert(1, DecompressionFilter())

        Pipeline(component_list).run()
