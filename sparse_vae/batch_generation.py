from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
from typing import *
import torch


# Quickly generate large numbers of batched samples from some language model. It's assumed that the model is on the GPU.
@torch.no_grad()
def batch_generate_samples(sample_func: Callable, num_samples: int, max_length: int, end_token: Optional[int]):
    # Allocate a big continuous block of pinned memory in the CPU that we can asynchronously copy GPU results into
    output_buffer = torch.empty(num_samples, max_length, device='cpu', dtype=torch.int16, pin_memory=True)
    pbar = tqdm(desc='Generating samples', smoothing=0.1, total=num_samples, unit='samples')

    # tokenizer = Tokenizer.from_file(str(Path.cwd() / 'sparse-vae-pretrained' / 'tokenizers' / 'yelp_review_full.json'))

    cur_idx = 0
    while cur_idx < num_samples:
        gpu_batch = sample_func().to(torch.int16)
        batch_size = gpu_batch.shape[0]

        output_buffer[cur_idx:cur_idx + batch_size].copy_(gpu_batch, non_blocking=True)
        cur_idx += batch_size

        pbar.update(n=batch_size)  # Let user know we've finished a batch right now

    pbar.close()

    # Trim away tokens after the [SEP] token
    print("Removing padding tokens...")
    end_indices = output_buffer.eq(end_token).nonzero(as_tuple=True)
    outputs = list(output_buffer)  # Splits the buffer into a list of 1D tensors, one for each sample

    for i, end in zip(*end_indices):
        end += 1
        if end < max_length:
            outputs[i] = outputs[i][:end]

    return outputs
