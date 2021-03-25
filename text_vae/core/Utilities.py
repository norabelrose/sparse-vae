from pathlib import Path
from torch import nn
from typing import *


class LambdaLayer(nn.Module):
    def __init__(self, function: Callable):
        super(LambdaLayer, self).__init__()
        self.function = function

    def forward(self, x: Any):
        return self.function(x)


# Get a new dictionary with a subset of (possibly transformed) values from the old dictionary. If you want a
# key-value pair to simply be copied from the old dict, list the key in `args`. If you want to rename a value,
# or get transformed values using arbitrary Python expressions, use **kwargs. Inspired by the R `dplyr` package.
T = TypeVar('T', bound=Mapping)
def transmute(big_dict: T, *args: str, **kwargs: str) -> T:
    new_dict = {k: big_dict[k] for k in args}
    eval_ctx = big_dict if isinstance(big_dict, dict) else dict(**big_dict)
    new_dict.update({new_k: eval(expr, eval_ctx) for new_k, expr in kwargs.items()})
    return new_dict


def get_checkpoint_path_for_name(experiment: str, ckpt_name: str) -> Path:
    ckpt_path = Path.cwd() / 'text-vae-logs' / experiment / ckpt_name / "checkpoints"
    try:
        # Open the most recent checkpoint
        ckpt = max(ckpt_path.glob('*.ckpt'), key=lambda file: file.lstat().st_mtime)
        return ckpt
    except ValueError:
        print(f"Couldn't find checkpoint at path {ckpt_path}")
        exit(1)
