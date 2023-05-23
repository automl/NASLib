import unittest
import logging
import torch
import os

from naslib.search_spaces import SimpleCellSearchSpace, NasBench301SearchSpace, HierarchicalSearchSpace, \
    NasBench201SearchSpace
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, DrNASOptimizer, RandomNASOptimizer
from naslib import utils
from naslib.utils import setup_logger
from naslib.search_spaces.core.primitives import Identity, SepConv, DilConv, Zero, MaxPool, AvgPool
from naslib.search_spaces.nasbench301.conversions import Genotype, convert_genotype_to_compact, \
    convert_compact_to_genotype, convert_genotype_to_config, convert_config_to_genotype, \
    convert_genotype_to_naslib, convert_naslib_to_genotype, get_cell_of_type

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
        self.optimizer = DARTSOptimizer(**config.search)
        self.optimizer.adapt_search_space(NasBench301SearchSpace(), config.dataset)
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529074, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529074, places=3)  # TODO: Improve this test

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.5546151, places=3)


class DartsGdasIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = GDASOptimizer(**config.search)
        self.optimizer.adapt_search_space(NasBench301SearchSpace(), config.dataset)
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        if torch.cuda.is_available():
            self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529, places=3)
            self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529, places=3)
        else:
            self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529072, places=3)
            self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529072, places=3)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.5546151, places=3)


class DartsDrNasIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = DrNASOptimizer(**config.search)
        self.optimizer.adapt_search_space(NasBench301SearchSpace(), config.dataset)
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        if torch.cuda.is_available():
            self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529, places=3)
            self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529, places=3)
        else:
            self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529074, places=3)
            self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529077, places=3)

    def test_feed_forward(self):
        final_arch = self.optimizer.get_final_architecture()
        logits = final_arch(data_train[0])
        self.assertTrue(logits.shape == (2, 10))
        self.assertAlmostEqual(logits[0, 0].detach().cpu().numpy(), 0.5546151, places=3)


class DartsRSWSIntegrationTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = RandomNASOptimizer(**config.search)
        self.optimizer.adapt_search_space(NasBench301SearchSpace(), config.dataset)
        self.optimizer.before_training()

    def test_update(self):
        stats = self.optimizer.step(data_train, data_val)
        self.assertTrue(len(stats) == 4)
        self.assertAlmostEqual(stats[2].detach().cpu().numpy(), 2.3529072, places=3)
        self.assertAlmostEqual(stats[3].detach().cpu().numpy(), 2.3529072, places=3)


class NasBench301SearchSpaceTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer_graph = DARTSOptimizer(**config.search)
        self.optimizer_subgraph = DARTSOptimizer(**config.search)
        self.graph = NasBench301SearchSpace()
        self.subgraph = self.graph.nodes[4]['subgraph']
        self.optimizer_graph.adapt_search_space(self.graph, config.dataset)
        self.optimizer_subgraph.adapt_search_space(self.subgraph, config.dataset)
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
            if isinstance(data['op'], list):
                for operation in data['op']:
                    self.assertIn(type(operation), [Identity, Zero, SepConv, DilConv, AvgPool, MaxPool])
            else:
                self.assertIn(type(data['op']), [Identity, Zero, SepConv, DilConv, AvgPool, MaxPool])

        # Check if the number of edges is the same as number of operations in the subgraph
        self.assertEqual(self.subgraph.number_of_edges(), self.num_ops)


class DartsConversionsTest(unittest.TestCase):

    def setUp(self):
        utils.set_seed(1)
        self.optimizer = DARTSOptimizer(**config.search)
        self.optimizer.graph = NasBench301SearchSpace()
        self.genotype = Genotype(
            normal=[
                ('max_pool_3x3', 0),
                ('avg_pool_3x3', 1),
                ('skip_connect', 0),
                ('sep_conv_3x3', 1),
                ('sep_conv_5x5', 1),
                ('dil_conv_3x3', 0),
                ('dil_conv_5x5', 0),
                ('max_pool_3x3', 2)],
            normal_concat=[2, 3, 4, 5],
            reduce=[
                ('max_pool_3x3', 0),
                ('avg_pool_3x3', 1),
                ('skip_connect', 2),
                ('sep_conv_3x3', 1),
                ('sep_conv_5x5', 0),
                ('sep_conv_5x5', 2),
                ('dil_conv_3x3', 2),
                ('dil_conv_5x5', 1)],
            reduce_concat=[2, 3, 4, 5])
        self.optimizer.before_training()

    def test_convert_genotype_to_compact_and_back(self):

        compact = convert_genotype_to_compact(self.genotype)
        genotype_from_compact = convert_compact_to_genotype(compact)

        assert self.genotype.normal == genotype_from_compact.normal
        assert self.genotype.reduce == genotype_from_compact.reduce

    def test_convert_genotype_to_config_and_back(self):
        config = convert_genotype_to_config(self.genotype)
        genotype_from_config = convert_config_to_genotype(config)
        config2 = convert_genotype_to_config(genotype_from_config)

        assert config == config2

    def test_convert_genotype_to_naslib(self):
        convert_genotype_to_naslib(self.genotype, self.optimizer.graph)
        normal_cell = get_cell_of_type(self.optimizer.graph, "normal_cell")
        reduction_cell = get_cell_of_type(self.optimizer.graph, "reduction_cell")

        normal_edges = {
            (1, 3): 'MaxPool',
            (2, 3): 'AvgPool',
            (1, 4): 'Identity',
            (2, 4): 'SepConv3x3',
            (1, 5): 'DilConv3x3',
            (2, 5): 'SepConv5x5',
            (1, 6): 'DilConv5x5',
            (3, 6): 'MaxPool',
            (3, 7): 'Identity',
            (4, 7): 'Identity',
            (5, 7): 'Identity',
            (6, 7): 'Identity'
        }

        reduction_edges = {
            (1, 3): 'MaxPool',
            (2, 3): 'AvgPool',
            (3, 4): 'Identity',
            (2, 4): 'SepConv3x3',
            (1, 5): 'SepConv5x5',
            (3, 5): 'SepConv5x5',
            (3, 6): 'DilConv3x3',
            (2, 6): 'DilConv5x5',
            (3, 7): 'Identity',
            (4, 7): 'Identity',
            (5, 7): 'Identity',
            (6, 7): 'Identity'
        }

        for edge, op_name in normal_edges.items():
            assert normal_cell.has_edge(*edge)
            assert normal_cell.edges[edge]['op'].get_op_name == op_name

        assert set(normal_cell.edges) == set(normal_edges.keys())

        for edge, op_name in reduction_edges.items():
            assert reduction_cell.has_edge(*edge)
            assert reduction_cell.edges[edge]['op'].get_op_name == op_name

        assert set(reduction_cell.edges) == set(reduction_edges.keys())

    def test_convert_genotype_to_naslib_and_back(self):
        convert_genotype_to_naslib(self.genotype, self.optimizer.graph)
        genotype = convert_naslib_to_genotype(self.optimizer.graph)

        def make_set_representation(edges):
            return [set([edges[i], edges[i + 1]]) for i in range(0, 8, 2)]

        genotype_normal = make_set_representation(genotype.normal)
        original_genotype_normal = make_set_representation(self.genotype.normal)

        assert genotype_normal == original_genotype_normal

        genotype_reduction = make_set_representation(genotype.reduce)
        original_genotype_reduction = make_set_representation(self.genotype.reduce)

        assert genotype_reduction == original_genotype_reduction


if __name__ == '__main__':
    unittest.main()
