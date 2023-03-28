import unittest
import torch
import numpy as np

from naslib.search_spaces import NasBench301SearchSpace
from naslib.search_spaces.core import Metric
from naslib.search_spaces.core.primitives import AbstractPrimitive


def create_dummy_api():
    class DummyNB301SurrogateModel:
        def predict(config, representation):
            return 96.0

    spec = (((0, 6), (1, 4), (0, 0), (1, 5), (1, 4), (3, 2), (0, 6), (3, 2)),
            ((0, 1), (1, 4), (0, 1), (1, 4), (2, 6), (3, 4), (1, 5), (2, 1)))

    data = {}
    data[spec] = {
        'runtime': [float(t * 100) for t in range(1, 99)],
        'train_losses': [float(l / 10) for l in range(98, 0, -1)],
        'val_accuracies': [float(i) for i in range(1, 99)]
    }

    api = {
        'nb301_arches': [spec],
        'nb301_model': DummyNB301SurrogateModel(),
        'nb301_data': data,
    }

    return api


def create_model():
    graph = NasBench301SearchSpace()
    spec = (((0, 1), (1, 4), (0, 6), (2, 0), (2, 6), (1, 4), (0, 5), (2, 2)),
            ((0, 5), (1, 6), (0, 1), (1, 5), (2, 5), (1, 2), (3, 3), (0, 3)))
    graph.set_spec(spec)
    return graph


class NasBench301SearchSpaceTest(unittest.TestCase):

    def test_set_and_get_spec(self):
        graph = NasBench301SearchSpace()
        spec = (((0, 1), (1, 4), (0, 6), (2, 0), (2, 6), (1, 4), (0, 5), (2, 2)),
                ((0, 5), (1, 6), (0, 1), (1, 5), (2, 5), (1, 2), (3, 3), (0, 3)))
        graph.set_spec(spec)
        retrieved_spec = graph.get_hash()

        self.assertEqual(spec, retrieved_spec)

    def test_set_spec_twice_with_instantiation(self):
        graph = NasBench301SearchSpace()
        spec = (((0, 1), (1, 4), (0, 6), (2, 0), (2, 6), (1, 4), (0, 5), (2, 2)),
                ((0, 5), (1, 6), (0, 1), (1, 5), (2, 5), (1, 2), (3, 3), (0, 3)))
        graph.set_spec(spec)
        retrieved_spec = graph.get_hash()

        self.assertEqual(spec, retrieved_spec)

        new_spec = (((0, 4), (1, 4), (0, 6), (2, 0), (2, 6), (1, 4), (0, 5), (2, 2)),
                    ((0, 4), (1, 4), (0, 1), (1, 5), (2, 5), (1, 2), (3, 3), (0, 3)))

        try:
            graph.set_spec(new_spec)
        except Exception as e:
            self.assertTrue(isinstance(e, AssertionError))
            exception_raised = True

        self.assertTrue(exception_raised)

    def test_set_spec_twice_without_instantiation(self):
        graph = NasBench301SearchSpace()
        graph.instantiate_model = False
        spec = (((0, 1), (1, 4), (0, 6), (2, 0), (2, 6), (1, 4), (0, 5), (2, 2)),
                ((0, 5), (1, 6), (0, 1), (1, 5), (2, 5), (1, 2), (3, 3), (0, 3)))
        graph.set_spec(spec)
        retrieved_spec = graph.get_hash()

        self.assertEqual(spec, retrieved_spec)

        new_spec = (((0, 4), (1, 4), (0, 6), (2, 0), (2, 6), (1, 4), (0, 5), (2, 2)),
                    ((0, 4), (1, 4), (0, 1), (1, 5), (2, 5), (1, 2), (3, 3), (0, 3)))
        graph.set_spec(new_spec)
        retrieved_spec = graph.get_hash()

        self.assertEqual(new_spec, retrieved_spec)

    def test_sample_random_architecture(self):
        graph = NasBench301SearchSpace()
        np.random.seed(9001)
        graph.sample_random_architecture()
        spec = graph.get_hash()
        spec_truth = (((0, 1), (1, 4), (0, 6), (2, 0), (2, 6), (1, 4), (0, 5), (2, 2)),
                      ((0, 5), (1, 6), (0, 1), (1, 5), (2, 5), (1, 2), (3, 3), (0, 3)))

        self.assertEqual(spec, spec_truth)

    def test_sample_random_architecture_from_labeled(self):
        graph = NasBench301SearchSpace()
        graph.sample_random_architecture(dataset_api=create_dummy_api(), load_labeled=True)
        spec = graph.get_hash()
        spec_truth = (((0, 6), (1, 4), (0, 0), (1, 5), (1, 4), (3, 2), (0, 6), (3, 2)),
                      ((0, 1), (1, 4), (0, 1), (1, 4), (2, 6), (3, 4), (1, 5), (2, 1)))

        self.assertEqual(spec, spec_truth)

    def test_forward_pass(self):
        graph = create_model()

        out = graph(torch.randn(3, 3, 32, 32))
        self.assertEqual(out.shape, (3, 10))

    def test_forward_pass_aux_head(self):
        graph = create_model()

        graph(torch.randn(3, 3, 32, 32))
        aux_out = graph.auxiliary_logits()
        self.assertEqual(aux_out.shape, (3, 512, 8, 8))

    def test_forward_pass_aux_head_eval(self):
        graph = create_model()
        graph.prepare_discretization()
        graph.prepare_evaluation()

        out = graph(torch.randn(3, 3, 32, 32))

        self.assertEqual(out.shape, (3, 10))

    def test_prepare_discretization(self):
        graph = create_model()

        graph.prepare_discretization()

        for n in range(4, 12):
            for e in graph.nodes[n]['subgraph'].edges(data=True):
                assert isinstance(e[2]['op'], AbstractPrimitive)

    def test_query_no_api(self):
        graph = NasBench301SearchSpace()
        graph.sample_random_architecture(dataset_api=create_dummy_api(), load_labeled=True)

        try:
            results = graph.query()
        except Exception as e:
            self.assertTrue(isinstance(e, NotImplementedError))

    def test_query_no_dataset(self):
        graph = NasBench301SearchSpace()
        graph.sample_random_architecture(dataset_api=create_dummy_api(), load_labeled=True)

        try:
            results = graph.query(dataset_api=create_dummy_api())
        except Exception as e:
            self.assertTrue(isinstance(e, AssertionError))

    def test_query(self):
        graph = NasBench301SearchSpace()
        graph.sample_random_architecture(dataset_api=create_dummy_api(), load_labeled=True)

        dummy_api = create_dummy_api()
        dummy_spec = list(dummy_api['nb301_data'].keys())[0]
        dummy_data = dummy_api['nb301_data'][dummy_spec]

        val_acc = graph.query(metric=Metric.VAL_ACCURACY, dataset_api=dummy_api)
        self.assertEqual(val_acc, dummy_data['val_accuracies'][-1])

        loss = graph.query(metric=Metric.TRAIN_LOSS, dataset_api=dummy_api)
        self.assertEqual(loss, dummy_data['train_losses'][-1])

        time = graph.query(metric=Metric.TRAIN_TIME, dataset_api=dummy_api)
        self.assertEqual(tuple(time), tuple(dummy_data['runtime']))

        val_accs = graph.query(metric=Metric.VAL_ACCURACY, dataset_api=dummy_api, full_lc=True)
        self.assertEqual(tuple(val_accs), tuple(dummy_data['val_accuracies']))

        loss = graph.query(metric=Metric.TRAIN_LOSS, dataset_api=dummy_api, full_lc=True)
        self.assertEqual(tuple(loss), tuple(dummy_data['train_losses']))

    def test_get_arch_iterator(self):
        graph = NasBench301SearchSpace()
        api = create_dummy_api()
        it = graph.get_arch_iterator(api)

        self.assertEqual(len(list(it)), 1)
        np.testing.assert_array_equal(list(it)[0], api['nb301_arches'][0])

    def test_mutate(self):
        graph_parent = create_model()
        graph_child = NasBench301SearchSpace()

        graph_child.mutate(graph_parent)

        parent_spec = graph_parent.get_hash()
        child_spec = graph_child.get_hash()

        self.assertNotEqual(parent_spec, child_spec)

        out = graph_child(torch.randn(3, 3, 32, 32))
        self.assertEqual(out.shape, (3, 10))

    # def test_get_nbhd(self):
    #     graph = create_model()
    #     neighbours = graph.get_nbhd()

    #     assert len(neighbours) == 120


if __name__ == '__main__':
    unittest.main()
