import unittest
import logging
import torch
import os

from naslib.search_spaces import SimpleCellSearchSpace, DartsSearchSpace, HierarchicalSearchSpace, NasBench201SeachSpace
from naslib.optimizers import DARTSOptimizer, GDASOptimizer
from naslib.utils import utils, setup_logger

logger = setup_logger(os.path.join("/tmp", "tests.log"))
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


class SimpleCellDartsIntegrationTest(unittest.TestCase):

    search_space = SimpleCellSearchSpace()
    optimizer = DARTSOptimizer(config)
    optimizer.adapt_search_space(search_space)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        data = torch.ones([2, 3, 32, 32])
        logits = final_arch(data)
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().numpy(), 0.092, places=3)


    def test_update(self):
        data_train = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
        data_val = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
        stats = self.optimizer.step(data_train, data_val)

        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().numpy(), 2.430, places=3)
        self.assertAlmostEqual(stats[3].detach().numpy(), 2.430, places=3)


class SimpleCellGdasIntegrationTest(unittest.TestCase):

    search_space = SimpleCellSearchSpace()
    optimizer = GDASOptimizer(config)
    optimizer.adapt_search_space(search_space)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        data = torch.ones([2, 3, 32, 32])
        logits = final_arch(data)
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().numpy(), -0.036, places=3)


    def test_update(self):
        data_train = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
        data_val = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
        stats = self.optimizer.step(data_train, data_val)

        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().numpy(), 2.230, places=3)
        self.assertAlmostEqual(stats[3].detach().numpy(), 2.230, places=3)


class DartsAllSearchSpacesTest(unittest.TestCase):

    search_spaces = [
        SimpleCellSearchSpace(), 
        DartsSearchSpace(), 
        HierarchicalSearchSpace(), 
        NasBench201SeachSpace(),
    ]

    def test_feed_forward(self):
        for sspace in self.search_spaces:
            print("feed forward", sspace.__class__)
            optimizer = DARTSOptimizer(config)
            optimizer.adapt_search_space(sspace)
            final_arch = optimizer.get_final_architecture()
            data = torch.ones([2, 3, 32, 32])
            logits = final_arch(data)
            self.assertTrue(logits.shape == (2, 10))


    def test_update(self):
        for sspace in self.search_spaces:
            print("update", sspace.__class__)
            optimizer = DARTSOptimizer(config)
            optimizer.adapt_search_space(sspace)
            data_train = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
            data_val = (torch.ones([2, 3, 32, 32]), torch.ones([2]).long())
            stats = optimizer.step(data_train, data_val)



if __name__ == '__main__':
    unittest.main()