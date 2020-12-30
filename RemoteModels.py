import os
import requests
import tarfile

from pathlib import Path
from tqdm.auto import tqdm

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
    content_length = response.headers.get('content-length')
    content_length = int(content_length) if content_length else None

    iterator = response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE)
    iterator = tqdm(iterator, desc="Downloading model", total=content_length, unit='MB')

    with local_path.open(mode='wb') as f:
        for chunk in iterator:
            f.write(chunk)

    print("Done. Now unzipping...")
    with tarfile.open(local_path, 'r:gz') as f:
        f.extractall(path=models_dir)
        os.remove(local_path)

    # The name of the tarfile but without the .tar.gz
    return models_dir / os.path.splitext(model_name)[0]
