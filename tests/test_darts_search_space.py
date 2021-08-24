import unittest
import logging
import torch
import os

from naslib.search_spaces import SimpleCellSearchSpace, DartsSearchSpace, HierarchicalSearchSpace, \
    NasBench201SearchSpace
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, DrNASOptimizer, RandomNASOptimizer
from naslib.utils import utils, setup_logger
from naslib.search_spaces.core.primitives import Identity, SepConv, DilConv, Zero, MaxPool, AvgPool

logger = setup_logger(os.path.join(utils.get_project_root().parent, "tmp", "tests.log"))
logger.handlers[0].setLevel(logging.FATAL)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class DartsDartsIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = DARTSOptimizer(config)
        self.optimizer.adapt_search_space(DartsSearchSpace())
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529074, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529074, places=3) #TODO: Improve this test

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.5091561, places=3)


class DartsGdasIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = GDASOptimizer(config)
        self.optimizer.adapt_search_space(DartsSearchSpace())
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        if torch.cuda.is_available():
            self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529, places=3)
            self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529, places=3)
        else:
            self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529074, places=3)
            self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529074, places=3)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.5091561, places=3)


class DartsDrNasIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = DrNASOptimizer(config)
        self.optimizer.adapt_search_space(DartsSearchSpace())
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        if torch.cuda.is_available():
            self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529, places=3)
            self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529, places=3)
        else:
            self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529074, places=3)
            self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529074, places=3)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.5091561, places=3)


class DartsRSWSIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = RandomNASOptimizer(config)
        self.optimizer.adapt_search_space(DartsSearchSpace())
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529074, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529074, places=3)


class DartsSearchSpaceTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer_graph = DARTSOptimizer(config)
        self.optimizer_subgraph = DARTSOptimizer(config)
        self.graph = DartsSearchSpace()
        self.subgraph = self.graph.nodes[4]['subgraph']
        self.optimizer_graph.adapt_search_space(self.graph)
        self.optimizer_subgraph.adapt_search_space(self.subgraph)
        self.optimizer_graph.before_training()
        self.optimizer_subgraph.before_training()
        self.num_ops = 0

    def test_update(self):
        # Check the total numbers of parameters and the number of edges of the graph
        self.assertEqual(self.optimizer_graph.get_model_size(), 1.930618)
        self.assertEqual(self.graph.number_of_edges(), 19)
        # Check the total numbers of parameters and the number of edges of the subgraph
        self.assertEqual(self.optimizer_subgraph.get_model_size(), 0.044352)
        self.assertEqual(self.subgraph.number_of_edges(), 18)
        # Check if the types of operations in the edges are the same as the primitive operations
        for _, _, data in self.subgraph.edges.data():
            self.num_ops += 1
            # Not all operations are lists of primitives, e.g. edges connecting to the output are always the identity
            if type(data['op']) == list:
                for operation in data['op']:
                    self.assertIn(type(operation), [Identity, Zero, SepConv, DilConv, AvgPool, MaxPool])
            else:
                self.assertIn(type(data['op']), [Identity, Zero, SepConv, DilConv, AvgPool, MaxPool])

        # Check if the number of edges is the same as number of operations in the subgraph
        self.assertEqual(self.subgraph.number_of_edges(), self.num_ops)


if __name__ == '__main__':
    unittest.main()
