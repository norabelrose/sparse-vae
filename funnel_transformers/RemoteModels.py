import os
import requests
import sys
import tarfile

from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict

DOWNLOAD_CHUNK_SIZE = 5242880  # 5 MB


# Returns the path to the local file once downloaded
def load_remote_model(url: str) -> Path:
    cache_dir = Path(os.getenv('XDG_CACHE_HOME', '~/.cache'))
    models_dir = cache_dir.expanduser() / 'text_vae' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if we've already downloaded it
    model_name = os.path.basename(url)
    local_path = models_dir / model_name
    if local_path.exists():
        return local_path

    response = requests.get(url, stream=True)
    response.raise_for_status()
    content_length = response.headers.get('content-length')
    if content_length:
        content_length = int(content_length)
        chunk_size = min(DOWNLOAD_CHUNK_SIZE, content_length // 1000)
    else:
        content_length = None
        chunk_size = DOWNLOAD_CHUNK_SIZE

    iterator = response.iter_content(chunk_size=chunk_size)
    pbar = tqdm(desc="Downloading model", total=content_length, unit='iB', unit_scale=True)

    try:
        with local_path.open(mode='wb') as f:
            for chunk in iterator:
                f.write(chunk)
                pbar.update(len(chunk))

        print("Done. Now unzipping...")
        with tarfile.open(local_path, 'r:gz') as f:
            f.extractall(path=models_dir)
    except KeyboardInterrupt:
        raise
    finally:
        os.remove(local_path)
        pbar.close()

    # The name of the tarfile but without the .tar.gz
    return models_dir / os.path.splitext(model_name)[0]


# Performs a series of replace operations on a string defined by a dictionary.
def replace_all(string: str, mappings: Dict[str, str]) -> str:
    for old, new in mappings:
        string = string.replace(old, new)

    return string
