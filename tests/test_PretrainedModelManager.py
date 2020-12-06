import unittest
from ..PretrainedModelManager import PretrainedModelManager
from ..Utilities import *

class TestPretrainedLoading(unittest.TestCase):
    @staticmethod
    def iterate_models() -> Iterator[Tuple[Tuple[int, ...], bool]]:
        layouts_to_test = PretrainedModelManager.block_size_to_name.keys()
        for layout in layouts_to_test:
            for get_tf in (True, False):
                yield layout, get_tf

    @classmethod
    def setUpClass(cls) -> None:
        for layout, get_tf in cls.iterate_models():
            PretrainedModelManager.download_model_for_block_layout(layout, include_generator=get_tf)

    def test_download(self):
        for layout, get_tf in self.iterate_models():
            path = PretrainedModelManager.path_to_cached_model_for_block_layout(layout, include_generator=get_tf)
            self.assertTrue(os.path.exists(path))

    def test_load_pytorch(self):
        for layout, get_tf in self.iterate_models():
            if get_tf:
                continue

            model = PretrainedModelManager.get_model(layout, get_tf)
            self.assertIsNotNone(model)

    def test_load_tensorflow(self):
        for layout, get_tf in self.iterate_models():
            if not get_tf:
                continue

            model = PretrainedModelManager.get_model(layout, get_tf)
            self.assertIsNotNone(model)

    @classmethod
    def tearDownClass(cls) -> None:
        for layout, get_tf in cls.iterate_models():
            path = PretrainedModelManager.path_to_cached_model_for_block_layout(layout, with_generator=get_tf)
            os.remove(path)

if __name__ == '__main__':
    unittest.main()
