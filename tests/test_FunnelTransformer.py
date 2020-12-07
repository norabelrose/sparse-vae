import re
import os
import torch
import unittest
from tqdm.auto import tqdm
from ..funnel_transformers.FunnelTransformer import FunnelConfig, FunnelTransformer
from ..PretrainedModelManager import PretrainedModelManager

# This should be set to wherever the 'pytorch' directory of original Funnel-Transformer package is on your system
PATH_TO_FUNNEL_TRANSFORMER_DIR = '~/Code/Python/Funnel-Transformer/pytorch/'


class TestFunnelTransformer(unittest.TestCase):
    # Test whether FunnelTransformer produces the same results as the original Funnel-Transformer
    # implementation (within a small error tolerance) if initialized from a pretrained checkpoint.
    @unittest.skipIf(not PATH_TO_FUNNEL_TRANSFORMER_DIR)
    def test_backward_compatibility(self):
        directory = os.path.expanduser(PATH_TO_FUNNEL_TRANSFORMER_DIR)
        if not os.path.isdir(directory):
            print(f"test_backward_compatibility: Can't find the Funnel-Transformer package on your system. Make sure"
                  f" you've downloaded it (see https://github.com/laiguokun/Funnel-Transformer) and then update "
                  f" the PATH_TO_FUNNEL_TRANSFORMER_DIR constant in {__file__}. Skipping test for now.")
            return

        # Load the ops.py file- it doesn't import any other Funnel-Transformer file and modeling.py needs it
        ops_file = os.path.join(directory, 'ops.py')
        with open(ops_file, 'r') as f:
            exec(f.read())

        # Now load the modeling.py file, but do surgery on it to remove the 'from ops import...' statements.
        # These cause runtime errors because 'ops' doesn't exist as a module from our perspective. They're also
        # unnecessary because we just directly loaded all the classes and functions from ops.py into global scope.
        modeling_file = os.path.join(directory, 'modeling.py')
        with open(modeling_file, 'r') as f:
            file_text = f.read()

            # Do surgery
            file_text = re.sub(r'from ops import [a-zA-Z]+\n', '', file_text)
            exec(file_text)  # Load everything

        new_model = PretrainedModelManager.get_model((4, 4, 4))
        old_config_dict = new_model.config.get_backward_compatible_dict()

        checkpoint_path = PretrainedModelManager.path_to_cached_model_for_block_layout((4, 4, 4))
        old_config = eval('ModelConfig(**old_config_dict)')
        old_model: torch.nn.Module = eval('FunnelFTM(old_config)')

        with open(checkpoint_path, 'rb') as f:
            old_model.load_state_dict(torch.load(f))

        print('Running 1000 forward passes of both models...')

        total_mse = 0.0
        for _ in tqdm(range(1000), unit='passes'):
            inputs = torch.randn(5, 512)

            output_new = new_model(inputs)
            output_old = old_model(inputs)
            total_mse += torch.mean((output_new - output_old) ** 2)

        loss = total_mse / 1000
        print('MSE: ', loss)
