from collections import OrderedDict
from typing import Any, List, Tuple
import torch
from numpy import prod
from torch import Tensor


# Literally copied from the collections module documentation- used to store CPU buffers
class LRUCache(OrderedDict):
    def __init__(self, maxsize=10, /, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


# Special version of the built-in PyTorch activation checkpointing function which
# automatically and asynchronously offloads activations to CPU memory in the forward
# pass and then loads them back into GPU memory just in time for the backward pass
class ActivationOffloadFunction(torch.autograd.Function):
    max_checkpoints = 5
    offloaded_checkpoints = []  # Used as a stack; allows for prefetching during the backward pass
    prefetched_checkpoint = None
    free_cpu_buffers = LRUCache()  # Keys are (numel, dtype) tuples, values are CPU tensors

    @classmethod
    def forward(cls, ctx, run_function, preserve_rng_state, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.had_autocast_in_fwd = torch.is_autocast_enabled()
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)

        with torch.no_grad():
            outputs = run_function(*args)

        # Don't offload the input activations if we've already hit the max
        if len(cls.offloaded_checkpoints) >= cls.max_checkpoints:
            ctx.did_offload = False
            ctx.save_for_backward(*args)
            return outputs

        buffers_used = []
        for arg in args:
            buffer = cls._pop_smallest_compatible_buffer(arg)
            if buffer is None:
                buffer = torch.empty_like(arg.flatten(), device='cpu', pin_memory=True)

            # Keep track of the shape of the tensor that this buffer is storing- buffers are all 1D
            buffer.payload_device = arg.device  # type: ignore
            buffer.payload_shape = arg.shape    # type: ignore

            # Asynchronously copy to CPU memory
            buffer[:arg.numel()].copy_(arg.flatten(), non_blocking=True)
            buffers_used.append(buffer)

        ctx.did_offload = True
        ctx.save_for_backward(*buffers_used)
        cls.offloaded_checkpoints.append(buffers_used)

        return outputs

    @classmethod
    def _pop_smallest_compatible_buffer(cls, data):
        # Return the smallest buffer that's big enough
        candidates = filter(lambda x: x[0] >= data.numel() and x[1] == data.dtype, cls.free_cpu_buffers)
        buf_key = min(candidates, key=lambda x: x[0], default=None)
        return cls.free_cpu_buffers.pop(buf_key) if buf_key else None

    @classmethod
    def _load_offloaded_checkpoint(cls, offloaded_ckpt: List[Tensor], non_blocking: bool) -> List[Tensor]:
        gpu_ckpt = []
        for buffer in offloaded_ckpt:
            payload_device = getattr(buffer, 'payload_device', None)
            payload_shape = getattr(buffer, 'payload_shape', None)
            assert payload_device is not None and payload_shape is not None, "Detected invalid offloaded autograd state"

            gpu_tensor = buffer[:prod(payload_shape)].to(payload_device, non_blocking=non_blocking)
            gpu_ckpt.append(gpu_tensor.view(*payload_shape))

            # Recycle the buffer for future backward passes
            cls.free_cpu_buffers[buffer.numel(), buffer.dtype] = buffer

        return gpu_ckpt

    @classmethod
    def backward(cls, ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")

        pt_ckpt = ctx.saved_tensors
        ckpt_stack = cls.offloaded_checkpoints

        if ctx.did_offload:
            stack_ckpt = ckpt_stack.pop()
            assert len(stack_ckpt) == len(pt_ckpt) and all(x.is_set_to(y) for x, y in zip(pt_ckpt, stack_ckpt)), \
                "Detected invalid offloaded autograd state"

            if cls.prefetched_checkpoint:
                inputs = cls.prefetched_checkpoint
                cls.prefetched_checkpoint = None
            else:
                inputs = cls._load_offloaded_checkpoint(stack_ckpt, non_blocking=False)
        else:
            inputs = pt_ckpt

        # Asynchronously prefetch offloaded state for the next operation
        if ckpt_stack and not cls.prefetched_checkpoint:
            torch.cuda.synchronize()    # Avoid prefetching too far ahead
            cls.prefetched_checkpoint = cls._load_offloaded_checkpoint(ckpt_stack[-1], non_blocking=True)

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(inputs)   # type: ignore
            with torch.enable_grad(), torch.cuda.amp.autocast(ctx.had_autocast_in_fwd):
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)
        return (None, None) + grads


def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list(set(arg.get_device() for arg in args if isinstance(arg, torch.Tensor) and arg.is_cuda))

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states) -> None:
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)

def offload(func, *args):
    return ActivationOffloadFunction.apply(func, True, *args)
