from dataclasses import dataclass
from torch import Tensor


# Super simple convenience class for making sure tensors that represent batches of variable-length
# sequences don't get separated from the padding masks that indicate where each sequence ends.
@dataclass
class PaddedTensor:
    data: Tensor
    padding: Tensor
