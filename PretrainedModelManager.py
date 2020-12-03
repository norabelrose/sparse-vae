import os
import re
import requests
import tarfile
import tensorflow as tf
from collections import OrderedDict
from tqdm.auto import tqdm
from typing import *

from .funnel_transformers.modeling import FunnelTransformer, FunnelConfig


class PretrainedModelManager:
    block_size_to_name: ClassVar[dict] = {
        (4, 4, 4): "B4-4-4H768-ELEC",
        (6, 6, 6): "B6-6-6H768-ELEC",
        (8, 8, 8): "B8-8-8H1024-ELEC",
        (10, 10, 10): "B10-10-10H1024-ELEC"
    }
    block_size_to_dims: ClassVar[dict] = {
        (4, 4, 4): (768, 12),  # d_model and num_heads
        (6, 6, 6): (768, 12),
        (8, 8, 8): (1024, 16),
        (10, 10, 10): (1024, 16)
    }

    @classmethod
    def model_name_for_block_layout(cls, block_layout: Tuple[int, ...],
                                    include_generator: bool = False) -> Optional[str]:
        if len(block_layout) < 3:
            return None

        funnel_layout = block_layout[0:3]
        if funnel_layout not in cls.block_size_to_name:
            return None

        # noinspection PyTypeChecker
        name = cls.block_size_to_name[funnel_layout]
        if include_generator:
            name += "-FULL"

        return name

    @classmethod
    def path_to_cached_model_for_block_layout(cls, block_layout: Tuple[int, ...],
                                              include_generator: bool = False) -> Optional[str]:
        name = cls.model_name_for_block_layout(block_layout, include_generator)
        if not name:  # Sanity check
            return None

        # Code copied from HuggingFace
        cache_dir = os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'))
        torch_cache_home = os.path.expanduser(os.getenv('TORCH_HOME', cache_dir, 'torch'))
        text_vae_home = os.path.join(torch_cache_home, 'text-vae')
        pretrained_dir = os.path.join(text_vae_home, 'pretrained-models')
        model_path = os.path.join(pretrained_dir, name)

        return model_path

    @classmethod
    def download_model_for_block_layout(cls, block_layout: Tuple[int, ...], include_generator: bool = False,
                                        use_tqdm: bool = True):
        name = cls.model_name_for_block_layout(block_layout, include_generator)

        # If we want to get the generator weights, we need to download the TensorFlow checkpoint and then convert it.
        name += "-TF" if include_generator else "-PT"

        url = f"http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/{name}.tar.gz"
        download_path = cls.path_to_cached_model_for_block_layout(block_layout, include_generator)

        with open(download_path, 'wb') as f:
            print(f"Downloading pretrained model from {url}...")

            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            iterator = response.iter_content(chunk_size=500000)

            if total_length is None and use_tqdm:
                print("Didn't get a content-length header, so we can't show a progress bar. Continuing...")
            elif use_tqdm:
                iterator = tqdm(iterator, desc="Downloading model")

            for chunk in iterator:
                f.write(chunk)

        print("Done. Now unzipping...")
        with tarfile.open(download_path, 'r:gz') as f:
            f.extractall(path=name)  # Extract into a directory with the name of the model
        os.remove(download_path)

        if include_generator:
            print("Done. Now converting the checkpoint from TensorFlow to PyTorch...")
            #convert_checkpoint()
        else:
            print("Finished.")

    @classmethod
    def get_model(cls, block_layout: Tuple[int, ...], include_generator: bool = False) -> \
            Union[FunnelTransformer, Tuple[FunnelTransformer, FunnelTransformer]]:
        path = cls.path_to_cached_model_for_block_layout(block_layout, include_generator)
        if path is None:
            raise ValueError(f'PretrainedModelManager: No pretrained model exists with this block layout.')

        if not os.path.exists(path):
            cls.download_model_for_block_layout(block_layout, include_generator)

        raise NotImplementedError

    @classmethod
    def _get_generator_and_model_from_tf_ckpt(cls, path: str, block_layout: Tuple[int, ...]) -> \
            Tuple[FunnelTransformer, FunnelTransformer]:
        reader = tf.train.load_checkpoint(path)
        var_list = tf.train.list_variables(path)

        # noinspection PyTypeChecker
        d_model, num_heads = cls.block_size_to_dims[block_layout]
        generator = FunnelTransformer(FunnelConfig(
            block_sizes=block_layout,
            d_model=d_model // 4,
            n_head=num_heads // 4
        ))
        model = FunnelTransformer(FunnelConfig(
            block_sizes=block_layout,
            d_model=d_model,
            n_head=num_heads
        ))
        layer_regex = re.compile('layer_([0-9]+)/')

        # Example: 'generator/encoder/layer_0/rel_attn/layer_norm/gamma' -> (0, 'attention.layer_norm.weight')
        def apply_tf_variable_to_model_if_needed(funnel: FunnelTransformer, key_path: str, tf_reader):
            if 'adam' in key_path:  # We don't care about optimizer states
                return

            result = layer_regex.match(key_path)
            if result is None:  # Shouldn't actually happen but just in case
                return

            tensor = tf_reader.get_tensor(key_path)

            layer_num = int(result.group(1))
            key_path = key_path[result.span()[1]:]

            tf_to_pt = OrderedDict([
                ('r/kernel', 'r_kernel'),     # This is implemented as a separate layer in TF but not in PyTorch
                ('/', '.'),
                ('rel_attn', 'attention'),
                ('_head', ''),                # e.g. 'q_head' -> 'q'
                ('kernel', 'weight'),
                ('beta', 'bias'),             # For layer norm
                ('gamma', 'weight')
            ])
            for old, new in tf_to_pt:
                key_path = key_path.replace(old, new)

            funnel.apply_weight_tensor_with_key_path(layer_num, key_path, tensor)

        for var_name, size in var_list:
            if var_name.startswith('generator/encoder/'):
                apply_tf_variable_to_model_if_needed(generator, var_name, reader)

            if var_name.startswith('model/encoder/'):
                apply_tf_variable_to_model_if_needed(model, var_name, reader)

        return generator, model
