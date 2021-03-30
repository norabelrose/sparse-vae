from torch import Tensor
from typing import *
import logging
import torch.nn.functional as F


_logger = logging.getLogger(__name__)

# Tensor subclass that ensures tensors that represent batches of variable-length sequences don't
# get separated from the padding masks that indicate where each sequence ends. All operations
# performed on PaddedTensors yield new PaddedTensors, and the padding mask gets propagated automatically.
# The smart property 'padding' will try to scale the original padding mask to the shape of the data tensor
# *on-the-fly*, or raise an error if it's unable to do so. Importantly, this allows downsampled PaddedTensors
# to "remember" the original, full-resolution versions of their padding masks, and automatically return them
# when the tensor is upsampled.
class PaddedTensor(Tensor):
    @classmethod
    def from_raw(cls, data: Tensor, padding: Optional[Tensor] = None) -> 'PaddedTensor':
        tensor = data.as_subclass(cls)
        tensor.padding = padding if padding is not None else data.eq(0)
        return tensor

    @classmethod
    def from_dict(cls, data: Dict[str, Tensor]) -> 'PaddedTensor':
        return cls.from_raw(data['data'], padding=data['padding'])

    def to_dict(self) -> Dict[str, Tensor]:
        return {'data': self.as_subclass(Tensor), 'padding': self.padding}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        results = super().__torch_function__(func, types, args, kwargs)

        if isinstance(results, PaddedTensor):
            results._padding = self._padding.to(results.device)

        # Should handle functions like topk which return a (named) tuple
        elif isinstance(results, Iterable):
            for x in results:
                if isinstance(x, PaddedTensor):
                    x._padding = self._padding.to(x.device)
        else:
            _logger.warning(f"A PaddedTensor was passed to the unsupported PyTorch function '{func}'. "
                            f"The padding mask will not be automatically propagated.")

        return results

    @property
    def padding(self) -> Optional[Tensor]:
        if self._padding is None:
            return None

        pad_shape, data_shape = self._padding.shape, self.shape
        assert pad_shape[0] == data_shape[0] and self._padding.ndim <= self.ndim,\
            f"Propagated padding mask of shape {pad_shape} cannot be reconciled with data tensor of shape {data_shape}"

        scaled_padding = self._padding
        for dim, (pad_size, data_size) in enumerate(zip(pad_shape, data_shape)):
            if pad_size == data_size:
                continue

            assert scaled_padding.ndim == 2, "Currently we can only auto-resize 2D padding masks"
            padding_dtype = scaled_padding.dtype
            scaled_padding = scaled_padding.unsqueeze(1).float()  # These functions expect [N, C, L] Tensors
            if pad_size < data_size:
                scaled_padding = F.interpolate(scaled_padding, size=data_size)
            elif pad_size > data_size:
                scaled_padding = F.adaptive_max_pool1d(scaled_padding.unsqueeze(1), output_size=data_size)

            scaled_padding = scaled_padding.squeeze(1).to(padding_dtype)
            break

        return scaled_padding

    @padding.setter
    def padding(self, value: Optional[Tensor]):
        if value is not None:
            assert value.ndim <= self.ndim, "Padding cannot have more dimensions than the tensor itself"
            for dim, (pad_size, data_size) in enumerate(zip(value.shape, self.shape)):
                assert pad_size == data_size, f"Padding size {pad_size} must match data size {data_size} at dim {dim}"

        self._padding = value.to(self.device)

    def __repr__(self):
        return f"PaddedTensor of shape {list(self.shape)}:\n{super().__repr__()}\nPadding:\n{list(self.padding.shape)}"
