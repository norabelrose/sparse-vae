# Sort of in order of how many internal dependencies each file has
from .batch_generation import *
from .core import select_best_gpu
from .core.conditional_gaussian import *
from .core.generation import *
from .core.transformer import *
from .core.language_model import *
from .core.continuous_autoencoder import *
from .lstm_vae import *
from .lstm_language_model import *
from .text_data_module import *
from .transformer_vae import *

# Useful utility to have
from pathlib import Path

def get_checkpoint_path_for_name(experiment: str, ckpt_name: str) -> Path:
    ckpt_path = Path.cwd() / 'sparse-vae-logs' / experiment / ckpt_name / "checkpoints"
    try:
        # Open the most recent checkpoint
        ckpt = max(ckpt_path.glob('*.ckpt'), key=lambda file: file.lstat().st_mtime)
        return ckpt
    except ValueError:
        print(f"Couldn't find checkpoint at path {ckpt_path}")
        exit(1)
