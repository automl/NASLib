import unittest

from naslib.base.preprocessor_base import *


class PreprocessorBaseTest(unittest.TestCase):

    def test_preprocessor_base(self):
        test_preprocessor = PreprocessorBase(hyperparameter_config=None)
        print(test_preprocessor.get_hyperparameter_search_space())


if __name__ == "__main__":
    unittest.main()
