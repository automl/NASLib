import unittest
import numpy as np
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api, utils

x_data = np.load('assets/nb201_test_set_x.npy', allow_pickle=True)
y_data = np.load('assets/nb201_test_set_y.npy', allow_pickle=True)
info_data = np.load('assets/nb201_test_set_info.npy', allow_pickle=True)
times_data = np.load('assets/nb201_test_set_times.npy', allow_pickle=True)
test_idx = int(np.ceil(len(x_data) * 0.7))
train_data = (x_data[:test_idx], y_data[:test_idx], info_data[:test_idx], times_data[:test_idx])
test_data = (x_data[test_idx:], y_data[test_idx:], info_data[test_idx:], times_data[test_idx:])
supported_search_spaces = {
    'nasbench201': NasBench201SearchSpace()
}


class PredictorsTest(unittest.TestCase):

    def setUp(self):
        self.args = utils.parse_args(args=['--config-file', 'assets/test_predictor.yaml'])
        self.config = utils.get_config_from_args(self.args, config_type='predictor')
        self.search_space = supported_search_spaces[self.config.search_space]
        utils.set_seed(self.config.seed)
        self.load_labeled = (True if self.config.search_space in ['darts', 'nlp'] else False)
        self.dataset_api = get_dataset_api(search_space='test')

    def test_configFile(self):
        self.assertEqual(self.config.search_space, 'nasbench201')
        self.assertEqual(self.config.dataset, 'cifar10')
        self.assertEqual(self.config.test_size, 10)
        self.assertEqual(self.config.train_size_single, 20)
        self.assertEqual(self.config.seed, 1000)
        self.assertEqual(self.config.uniform_random, 1)

if __name__ == '__main__':
    unittest.main()
