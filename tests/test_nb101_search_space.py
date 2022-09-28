import unittest
import torch
import numpy as np

from naslib.search_spaces import NasBench101SearchSpace
from naslib.search_spaces.core import Metric

SPEC=(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 3, 3, 3, 1)
SAMPLED_SPEC=(0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 3, 3, 1)

def create_dummy_api():
    # TODO: Complete
    api = None
    return api

def create_model(n_classes=10):
    graph = NasBench101SearchSpace(n_classes=n_classes)
    graph.set_spec(SPEC)
    return graph


class NasBench101SearchSpaceTest(unittest.TestCase):

    def test_set_and_get_spec(self):
        graph = NasBench101SearchSpace()
        graph.set_spec(SPEC)
        retrieved_spec = graph.get_hash()

        assert SPEC == retrieved_spec

    def test_forward_pass(self):
        graph = create_model(n_classes=10)

        out = graph(torch.randn(3, 3, 32, 32))
        assert out.shape == (3, 10)

    # TODO: Complete. These tests require a dummy NAS-Bench-101 API.
    # def test_sample_random_architecture(self):
    #     graph = NasBench101SearchSpace()
    #     np.random.seed(9001)
    #     graph.sample_random_architecture()
    #     spec = graph.get_hash()

    #     assert spec == SAMPLED_SPEC

    # def test_query(self):
    #     graph = NasBench101SearchSpace()
    #     graph.set_spec(SPEC)


    # def test_get_arch_iterator(self):
    #     graph = NasBench101SearchSpace()
    #     it = graph.get_arch_iterator()

    #     archs = set(it)

    #     assert len(archs) == 423624
    #     assert '00005c142e6f48ac74fdcf73e3439874' in archs

    # def test_mutate(self):
    #     graph_parent = create_model()
    #     graph_child = NasBench101SearchSpace()

    #     graph_child.mutate(graph_parent)

    #     parent_spec = graph_parent.get_hash()
    #     child_spec = graph_child.get_hash()

    #     assert parent_spec != child_spec

    #     out = graph_child(torch.randn(3, 3, 32, 32))
    #     assert out.shape == (3, 10)

    # def test_get_nbhd(self):
    #     graph = create_model()
    #     neighbours = graph.get_nbhd()

    #     print(len(neighbours))
    #     assert len(neighbours) == 24
