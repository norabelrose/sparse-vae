# Uses pynvml to select the index of the GPU with the most free memory
def select_best_gpu() -> int:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo
    )
    from numpy import argmax

    nvmlInit()
    num_gpus = nvmlDeviceGetCount()
    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]
    free_memory = [nvmlDeviceGetMemoryInfo(handle).free for handle in handles]

    best_idx = argmax(free_memory)
    print(f"Selected GPU {best_idx}.")
    return int(best_idx)
