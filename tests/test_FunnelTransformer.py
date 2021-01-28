import re
import os
import torch
import unittest
from contextlib import nullcontext
from pytorch_lightning.trainer import seed_everything
from pytorch_lightning.utilities import AttributeDict
from text_vae import FunnelTransformer, FunnelTransformerHparams

# This should be set to wherever the 'pytorch' directory of original Funnel-Transformer package is on your system
TEXT_VAE_PATH_TO_FUNNEL_TRANSFORMERS = os.getenv("TEXT_VAE_PATH_TO_FUNNEL_TRANSFORMERS")


class TestFunnelTransformer(unittest.TestCase):
    # Test whether FunnelTransformer produces the same results as the original Funnel-Transformer
    # implementation (within a small error tolerance) if initialized from a pretrained checkpoint.
    @unittest.skipIf(not TEXT_VAE_PATH_TO_FUNNEL_TRANSFORMERS, "Path to Funnel-Transformers env variable not set")
    def test_backward_compatibility(self):
        directory = os.path.expanduser(TEXT_VAE_PATH_TO_FUNNEL_TRANSFORMERS)
        if not os.path.isdir(directory):
            print(f"test_backward_compatibility: Can't find the Funnel-Transformer package at {directory}. Make sure"
                  f" you've downloaded it (see https://github.com/laiguokun/Funnel-Transformer) and then set the"
                  f" TEXT_VAE_PATH_TO_FUNNEL_TRANSFORMERS environment variable appropriately. Skipping test for now.")
            return

        seed_everything(7295)

        # Load the ops.py file- it doesn't import any other Funnel-Transformer file and
        # modeling.py needs it
        ops_file = os.path.join(directory, 'ops.py')
        with open(ops_file, 'r') as f:
            exec(f.read(), globals())

        # Now load the modeling.py file, but do surgery on it to remove the 'from ops import...' statements.
        # These cause runtime errors because 'ops' doesn't exist as a module from our perspective. They're also
        # unnecessary because we just directly loaded all the classes and functions from RelativePositionalAttention.py
        # into global scope.
        modeling_file = os.path.join(directory, 'modeling.py')
        with open(modeling_file, 'r') as f:
            file_text = f.read()

            # Do surgery
            file_text = re.sub(r'from ops import [a-zA-Z]+\n', '', file_text)
            exec(file_text, globals())  # Load everything
        
        with torch.cuda.device(0) if torch.cuda.is_available() else nullcontext(), torch.no_grad():
            for positional_encoding_type in ("factorized", "rel_shift"):
                new_model = FunnelTransformer(FunnelTransformerHparams(
                    positional_encoding_type=positional_encoding_type,
                    return_block_outputs=True
                ))
                new_model.load_pretrained_weights()
                new_model.eval()
                
                new_config = new_model.hparams
                old_config_dict = AttributeDict(**new_model.get_backward_compatible_dict())
                old_model_args = AttributeDict(**new_model.get_backward_compatible_args())
        
                checkpoint_path = new_model.path_to_pretrained_checkpoint() / "model.pt"
                old_config = eval('ModelConfig(**old_config_dict)')
                old_model: torch.nn.Module = eval('FunnelTFM(old_config, old_model_args, cls_target=False)')
                old_model.eval()  # Turn off dropout
        
                with open(checkpoint_path, 'rb') as f:
                    old_model.load_state_dict(torch.load(f))
        
                print('Running forward pass of both models...')
        
                # Tokens below 999 are either unused or are special tokens like [CLS] and [MASK]
                inputs = torch.randint(low=999, high=new_config.vocab_size, size=(1, 512))
                
                output_old = old_model(inputs)
                output_new = new_model({'input': inputs})
                mean_err = torch.mean(abs(output_old[0][-1] - output_new['output']))
        
                print('Mean absolute error: ', mean_err.item())
                print('Old output: ', output_old[0][4::4])
                print('New output: ', output_new)
                with self.subTest(positional_encoding_type):
                    self.assertTrue(mean_err < 1e-4)
