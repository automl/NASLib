import unittest
import logging
import torch
import os

from naslib.search_spaces import HierarchicalSearchSpace
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, DrNASOptimizer
from naslib.search_spaces.core.primitives import Zero1x1, Identity, MaxPool1x1, AvgPool1x1, SepConv
from naslib.search_spaces.hierarchical.primitives import ConvBNReLU, DepthwiseConv
from naslib import utils
from naslib.utils import setup_logger

logger = setup_logger(os.path.join(utils.get_project_root().parent, "tmp", "tests.log"))
logger.handlers[0].setLevel(logging.FATAL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

data_train = (torch.ones([2, 3, 32, 32]).to(device), torch.ones([2]).to(device).long())
data_val = (torch.ones([2, 3, 32, 32]).to(device), torch.ones([2]).to(device).long())


class HierarchicalDartsIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = DARTSOptimizer(**config.search)
        self.optimizer.adapt_search_space(HierarchicalSearchSpace(), config.dataset)
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.4094, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.4094, places=3)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), -0.0545, places=3)


class HierarchicalGdasIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = GDASOptimizer(**config.search)
        self.optimizer.adapt_search_space(HierarchicalSearchSpace(), config.dataset)
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.4094, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.4094, places=3)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), -0.0545, places=3)


class HierarchicalDrNasIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = DrNASOptimizer(**config.search)
        self.optimizer.adapt_search_space(HierarchicalSearchSpace(), config.dataset)
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.4094, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.4094, places=3)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), -0.0545, places=3)


class HierarchicalSearchSpaceTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer_graph = DARTSOptimizer(**config.search)
        self.optimizer_subgraph = DARTSOptimizer(**config.search)
        self.graph = HierarchicalSearchSpace()
        self.subgraph = self.graph.edges[4, 5]['op']
        self.optimizer_graph.adapt_search_space(self.graph, config.dataset)
        self.optimizer_subgraph.adapt_search_space(self.subgraph, config.dataset)
        self.num_ops = 0

    def test_update(self):
        # Check the total numbers of parameters and the number of edges of the graph
        self.assertEqual(self.optimizer_graph.get_model_size(), 7.891546)
        self.assertEqual(self.graph.number_of_edges(), 7)
        # Check the total numbers of parameters and the number of edges of the subgraph
        self.assertEqual(self.optimizer_subgraph.get_model_size(), 1.60512)
        self.assertEqual(self.subgraph.number_of_edges(), 10)
        for _, _, data in self.subgraph.edges.data():
            self.num_ops += 1
            for _, _, data_ in data['op'][0].edges.data():
                for operation in data_['op']:
                    self.assertIn(type(operation), [Identity, Zero1x1, DepthwiseConv, ConvBNReLU, AvgPool1x1,
                                                    MaxPool1x1, SepConv])

        # Check if the number of edges is the same as number of operations in the subgraph
        self.assertEqual(self.subgraph.number_of_edges(), self.num_ops)


if __name__ == '__main__':
    unittest.main()
