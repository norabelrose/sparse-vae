import os
import requests
import tarfile

from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict

DOWNLOAD_CHUNK_SIZE = 5242880  # 5 MB


def remote_model_url_for_hparams(hparams, suffix: str="") -> str:
    block_size_to_name = {
        (4, 4, 4): "B4-4-4H768-ELEC",
        (6, 6, 6): "B6-6-6H768-ELEC",
        (8, 8, 8): "B8-8-8H1024-ELEC",
        (10, 10, 10): "B10-10-10H1024-ELEC"
    }
    block_size_to_dims = {
        (4, 4, 4): (768, 12),  # d_model and num_heads
        (6, 6, 6): (768, 12),
        (8, 8, 8): (1024, 16),
        (10, 10, 10): (1024, 16)
    }

    # Sanity checks
    beginning_blocks = hparams.block_sizes[0:3]
    pretrained_d_model = block_size_to_dims[beginning_blocks][0]
    assert len(hparams.block_sizes) >= 3
    assert beginning_blocks in block_size_to_name, f"No pretrained model with block layout {beginning_blocks}"
    assert hparams.d_model == pretrained_d_model, \
        f"Pretrained model {block_size_to_name[beginning_blocks]} requires d_model == {pretrained_d_model}"

    name = block_size_to_name[beginning_blocks] + suffix
    return f"http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/{name}.tar.gz"

# Returns the path to the local file once downloaded
def load_remote_model(url: str) -> Path:
    cache_dir = Path(os.getenv('XDG_CACHE_HOME', '~/.cache'))
    models_dir = cache_dir.expanduser() / 'text_vae' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if we've already downloaded it
    model_name = os.path.basename(url)
    local_path = models_dir / model_name.replace(".tar.gz", "")
    
    if local_path.exists():
        return local_path

    print(f"Downloading model from {url}...")

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

    archive_path = models_dir / model_name
    try:
        with archive_path.open(mode='wb') as f:
            for chunk in iterator:
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.close()

        print("Done. Now unzipping...")
        with tarfile.open(archive_path, 'r:gz') as f:
            f.extractall(path=models_dir)
    except KeyboardInterrupt:
        raise
    finally:
        os.remove(archive_path)

    return local_path


# Performs a series of replace operations on a string defined by a dictionary.
def replace_all(string: str, mappings: Dict[str, str]) -> str:
    for old, new in mappings.items():
        string = string.replace(old, new)

    return string
