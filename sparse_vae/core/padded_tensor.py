from torch import Tensor
from typing import *
import logging


_logger = logging.getLogger(__name__)

# Tensor subclass that ensures tensors that represent batches of variable-length sequences don't
# get separated from the padding masks that indicate where each sequence ends. All operations
# performed on PaddedTensors yield new PaddedTensors, and the padding mask gets propagated automatically.
class PaddedTensor(Tensor):
    @classmethod
    def from_raw(cls, data: Tensor, padding: Optional[Tensor] = None) -> 'PaddedTensor':
        tensor = data.as_subclass(cls)
        tensor.padding = padding if padding is not None else data.eq(0)
        return tensor

    @classmethod
    def unpadded(cls, data: Tensor) -> 'PaddedTensor':
        tensor = data.as_subclass(cls)
        tensor._padding = None
        return tensor

    def as_raw(self) -> Tensor:
        return self.as_subclass(Tensor)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        results = super().__torch_function__(func, types, args, kwargs)

        if results == NotImplemented:
            raw_kwargs = {k: v.as_raw() if isinstance(v, PaddedTensor) else v for k, v in kwargs.items()} if kwargs else {}
            raw_args = [arg.as_raw() if isinstance(arg, PaddedTensor) else arg for arg in args]
            results = func(*raw_args, **raw_kwargs)
            if isinstance(results, Tensor):
                results = self.unpadded(results)
                results._padding = self._padding
                return results
            else:
                _logger.warning(f"A PaddedTensor was passed to the function '{func}' in an unsupported fashion. Types"
                                f"of parameters were '{types}'. The padding mask will not be propagated.")

        elif isinstance(results, PaddedTensor):
            results._padding = self._padding if self._padding is not None else None

        # Should handle functions like topk which return a (named) tuple
        elif isinstance(results, Iterable):
            for x in results:
                if isinstance(x, PaddedTensor):
                    x._padding = self._padding if self._padding is not None else None

        return results

    @property
    def padding(self) -> Optional[Tensor]:
        if (padding := self._padding) is None:
            return None

        # Do this lazily to avoid unnecessary blocking device transfers with activation offloading
        if padding.device != self.device:
            padding = self._padding = padding.to(self.device)

        # Expand across batch dimension
        pad_shape, data_shape = self._padding.shape, self.shape
        if pad_shape[0] != data_shape[0] and pad_shape[0] == 1:
            return padding.expand(data_shape[0], *pad_shape[1:])

        # If the padding mask doesn't fit the data, just return None
        return padding if pad_shape[-1] == data_shape[padding.ndim - 1] else None

    @padding.setter
    def padding(self, value: Optional[Tensor]):
        if value is not None:
            assert value.ndim <= self.ndim, "Padding cannot have more dimensions than the tensor itself"
            for dim, (pad_size, data_size) in enumerate(zip(value.shape, self.shape)):
                assert pad_size == data_size, f"Padding size {pad_size} must match data size {data_size} at dim {dim}"

        self._padding = value.to(self.device) if value is not None else None

    def __repr__(self):
        padding_repr = list(self._padding.shape) if self._padding is not None else "(None)"
        return f"PaddedTensor of shape {list(self.shape)}:\n{super().__repr__()}\nPadding:\n{padding_repr}"
