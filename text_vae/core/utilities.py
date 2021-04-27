from pathlib import Path
from torch import nn
from typing import *


class LambdaLayer(nn.Module):
    def __init__(self, function: Callable):
        super(LambdaLayer, self).__init__()
        self.function = function

    def forward(self, x: Any):
        return self.function(x)


def get_checkpoint_path_for_name(experiment: str, ckpt_name: str) -> Path:
    ckpt_path = Path.cwd() / 'sparse-vae-logs' / experiment / ckpt_name / "checkpoints"
    try:
        # Open the most recent checkpoint
        ckpt = max(ckpt_path.glob('*.ckpt'), key=lambda file: file.lstat().st_mtime)
        return ckpt
    except ValueError:
        print(f"Couldn't find checkpoint at path {ckpt_path}")
        exit(1)
