import unittest
from shutil import rmtree
from ..PretrainedModelManager import PretrainedModelManager
from ..Utilities import *

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

    #@classmethod
    #def tearDownClass(cls) -> None:
    #    for layout, get_tf in cls.iterate_models():
    #        path = PretrainedModelManager.path_to_cached_model_for_block_layout(layout, include_generator=get_tf)
    #        rmtree(path)

if __name__ == '__main__':
    unittest.main()
