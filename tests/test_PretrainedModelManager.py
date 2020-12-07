import unittest
from ..PretrainedModelManager import PretrainedModelManager

class TestPretrainedLoading(unittest.TestCase):
    def test_load_pytorch(self):
        for layout in PretrainedModelManager.block_size_to_name.keys():
            model = PretrainedModelManager.get_model(layout, include_generator=False, strict=True)
            self.assertIsNotNone(model)

    def test_load_tensorflow(self):
        for layout in PretrainedModelManager.block_size_to_name.keys():
            model = PretrainedModelManager.get_model(layout, include_generator=True, strict=True)
            self.assertIsNotNone(model)
            self.assertEqual(len(model), 2)  # Make sure we have both the generator and discriminator

if __name__ == '__main__':
    unittest.main()
