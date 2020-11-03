import unittest
import logging
import torch
import os

from naslib.search_spaces import SimpleCellSearchSpace, DartsSearchSpace, HierarchicalSearchSpace, NasBench201SeachSpace
from naslib.optimizers import DARTSOptimizer, GDASOptimizer
from naslib.utils import utils, setup_logger

logger = setup_logger(os.path.join("tmp", "tests.log"))
logger.handlers[0].setLevel(logging.FATAL)
utils.set_seed(1)


config = utils.AttrDict()
config.dataset = 'cifar10'
config.search = utils.AttrDict()
config.search.grad_clip = None
config.search.learning_rate = 0.01
config.search.momentum = 0.1
config.search.weight_decay = 0.1
config.search.arch_learning_rate = 0.01
config.search.arch_weight_decay = 0.1
config.search.tau_max = 10
config.search.tau_min = 1
config.search.epochs = 2


class DartsDartsIntegrationTest(unittest.TestCase):

    search_space = DartsSearchSpace()
    optimizer = DARTSOptimizer(config)
    optimizer.adapt_search_space(search_space)

    def test_1update(self):
        data_train = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
        data_val = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
        stats = self.optimizer.step(data_train, data_val)

        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().numpy(), 2.8412, places=3)   # 2.3563
        self.assertAlmostEqual(stats[3].detach().numpy(), 2.8638, places=3)   # 2.3799
    

    def test_2feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        data = torch.ones([2, 3, 32, 32])
        logits = final_arch(data)
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().numpy(), -0.1624, places=3)   # -0.2567


class DartsGdasIntegrationTest(unittest.TestCase):

    search_space = DartsSearchSpace()
    optimizer = GDASOptimizer(config)
    optimizer.adapt_search_space(search_space)

    def test_1update(self):
        data_train = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
        data_val = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
        stats = self.optimizer.step(data_train, data_val)

        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().numpy(), 2.0730, places=3)
        self.assertAlmostEqual(stats[3].detach().numpy(), 2.3429, places=3)
    

    def test_2feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        data = torch.ones([2, 3, 32, 32])
        logits = final_arch(data)
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().numpy(), -0.3911, places=3)

if __name__ == '__main__':
    unittest.main()