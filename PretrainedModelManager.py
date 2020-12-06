import logging
import requests
import tarfile
import tensorflow as tf
from tqdm.auto import tqdm
from pathlib import Path

from .funnel_transformers.FunnelTransformer import *

ElectraModel = NewType('ElectraModel', Tuple[FunnelTransformer, FunnelTransformer])
PretrainedModel = NewType('PretrainedModel', Union[FunnelTransformer, ElectraModel])


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
    def get_model(cls, block_layout: Tuple[int, ...], include_generator: bool = False) -> PretrainedModel:
        path = cls.path_to_cached_model_for_block_layout(block_layout, include_generator)
        if path is None:
            raise ValueError(f'PretrainedModelManager: No pretrained model exists with this block layout.')

        if not os.path.exists(path):
            cls.download_model_for_block_layout(block_layout, include_generator)

        if include_generator:
            return cls._get_model_from_pt_ckpt(path, block_layout)
        else:
            return cls._get_generator_and_model_from_tf_ckpt(path, block_layout)

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
                                              include_generator: bool = False) -> Optional[Path]:
        name = cls.model_name_for_block_layout(block_layout, include_generator)
        if not name:  # Sanity check
            return None

        # Code copied from HuggingFace
        cache_dir = os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'))
        torch_cache_home = os.path.expanduser(os.getenv('TORCH_HOME', cache_dir))
        text_vae_home = os.path.join(torch_cache_home, 'text-vae')
        pretrained_dir = os.path.join(text_vae_home, 'pretrained-models')
        model_path = os.path.join(pretrained_dir, name)

        return Path(model_path)

    @classmethod
    def download_model_for_block_layout(cls, block_layout: Tuple[int, ...], include_generator: bool = False,
                                        use_tqdm: bool = True):
        name = cls.model_name_for_block_layout(block_layout, include_generator)

        # If we want to get the generator weights, we need to download the TensorFlow checkpoint and then convert it.
        name += "-TF" if include_generator else "-PT"

        url = f"http://storage.googleapis.com/funnel-transformer/funnel_ckpts_all/{name}.tar.gz"
        download_path = cls.path_to_cached_model_for_block_layout(block_layout, include_generator)
        download_path.parent.mkdir(parents=True, exist_ok=True)  # Make sure parent dir exists
        download_path = download_path.with_suffix('.tar.gz')

        with open(download_path, 'wb') as f:
            print(f"Downloading pretrained model from {url}...")

            response = requests.get(url, stream=True)
            total_length = int(response.headers.get('content-length'))
            iterator = response.iter_content(chunk_size=1048576)

            if total_length is None and use_tqdm:
                print("Didn't get a content-length header, so we can't show a progress bar. Continuing...")
            elif use_tqdm:
                pbar = tqdm(desc="Downloading model", total=int(total_length), unit='bytes')

            for chunk in iterator:
                f.write(chunk)

                if use_tqdm:
                    pbar.update(len(chunk))

            if use_tqdm:
                pbar.close()

        print("Done. Now unzipping...")
        with tarfile.open(download_path, 'r:gz') as f:
            f.extractall(path=name)  # Extract into a directory with the name of the model
        os.remove(download_path)

        print("Done.")

    @classmethod
    def _get_model_from_pt_ckpt(cls, path: Path, block_layout: Tuple[int, ...]) -> FunnelTransformer:
        # noinspection PyTypeChecker
        d_model, num_heads = cls.block_size_to_dims[block_layout]
        model = FunnelTransformer(FunnelConfig(
            block_sizes=block_layout,
            d_model=d_model,
            n_head=num_heads
        ))

        # Our parameter names will look like this: 'blocks.0.layers.2.attention.v_head.bias', but the pretraining
        # files will have the form 'attn_layers.2.v_head.bias'. We need to convert here.
        state_dict = torch.load(str(path))
        noninitialized_keys = []

        for var_name, param, absolute_index in model.enumerate_parameters_by_layer():
            keys = var_name.split('.')
            keys[0] = replace_all(keys[0], [  # attention.v_head.bias -> attn_layers.v_head.bias
                ('attention', 'attn_layers'),
                ('feedforward', 'pffn_layers')
            ])

            keys.insert(1, str(absolute_index))  # attn_layers.v_head.bias -> attn_layers.2.v_head.bias
            old_name = '.'.join(keys)

            try:
                param.data = state_dict[old_name]
            except KeyError:
                noninitialized_keys.append({'new_name': var_name, 'old_name': old_name})

        if len(noninitialized_keys) > 0:
            logger = logging.getLogger(__name__)
            logger.warning(f'PretrainedModelManager: Failed to initialize weights: {noninitialized_keys}')

        return model

    @classmethod
    def _get_generator_and_model_from_tf_ckpt(cls, path: Path, block_layout: Tuple[int, ...]) -> ElectraModel:
        print("Loading model from TensorFlow checkpoint...")
        reader = tf.train.load_checkpoint(str(path))

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

        def convert_pt_parameter_name_to_tf(key_string: str, abs_index: int, prefix: str) -> str:
            # 'attention.v_head.bias' -> 'layer_2/rel_attn/v/bias'

            # 'feedforward.pffn.0.bias' -> 'feedforward.layer_1.bias'
            # 'feedforward.layer_norm.weight' -> 'feedforward.layer_norm.gamma'
            key_string = replace_all(key_string, [
                ('layer_norm.weight', 'layer_norm.gamma'),
                ('layer_norm.bias', 'layer_norm.beta'),
                ('pffn.0', 'layer_1'),
                ('pffn.3', 'layer_2'),
                ('r_kernel', 'r.kernel'),   # r_kernel is a parameter of a separate Dense layer in TF
                ('weight', 'kernel')
            ])

            keys = key_string.split('.')
            keys.insert(0, "layer_" + str(abs_index))

            keys[1] = replace_all(keys[1], [    # 'layer_17.feedforward.layer_1.bias' -> 'layer_17.ff.layer_1.bias'
                ('attention', 'rel_attn'),
                ('feedforward', 'ff')
            ])
            keys[2] = replace_all(keys[2], [    # 'layer_17.rel_attn.v_head.bias' -> 'layer_17.rel_attn.v.bias'
                ('_head', ''),
                ('post_proj', 'o')
            ])

            # 'layer_17.rel_attn.v.bias' -> 'model/encoder/layer_17/rel_attn/v/bias'
            return prefix + '/'.join(keys)

        for var_name, param, absolute_index in model.enumerate_parameters_by_layer():
            tf_name = convert_pt_parameter_name_to_tf(var_name, absolute_index, 'model/encoder/')
            param.data = reader.get_tensor(tf_name)

        for var_name, param, absolute_index in model.enumerate_parameters_by_layer():
            tf_name = convert_pt_parameter_name_to_tf(var_name, absolute_index, 'generator/encoder/')
            param.data = reader.get_tensor(tf_name)

        print("Finished.")
        return generator, model
