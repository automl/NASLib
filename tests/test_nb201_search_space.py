import unittest
import torch
import numpy as np

from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core import Metric
from naslib.search_spaces.core.primitives import AbstractPrimitive


def create_dummy_api():
    api = {
        'nb201_data': {
            '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|': {
                'cifar10-valid': {
                    'train_losses': [float(i) for i in range(199, 0, -1)],
                    'eval_losses': [float(i) for i in range(299, 100, -1)],
                    'train_acc1es': [float(i/2) for i in range(0, 200, 1)],
                    'eval_acc1es': [float(i/3) for i in range(0, 200, 1)],
                    'cost_info': {
                        'flops': 15.64737,
                        'params': 0.129306,
                        'latency': 0.0139359758611311,
                        'train_time': 7.221092998981476
                    }
                }
            }
        }
    }

    return api

def create_model(n_classes=10):
    graph = NasBench201SearchSpace(n_classes=n_classes)
    spec = (2, 2, 3, 4, 3, 2)
    graph.set_spec(spec)
    return graph


class NasBench201SearchSpaceTest(unittest.TestCase):

    def test_set_and_get_spec(self):
        graph = NasBench201SearchSpace()
        spec = (2, 2, 3, 4, 3, 2)
        graph.set_spec(spec)
        retrieved_spec = graph.get_hash()

        assert spec == retrieved_spec

    def test_sample_random_architecture(self):
        graph = NasBench201SearchSpace()
        np.random.seed(9001)
        graph.sample_random_architecture()
        spec = graph.get_hash()
        spec_truth = (1, 4, 0, 1, 3, 4)

        assert spec == spec_truth

    def test_forward_pass(self):
        graph = create_model(n_classes=10)

        out = graph(torch.randn(3, 3, 32, 32))
        assert out.shape == (3, 10)

        graph = create_model(n_classes=100)

        out = graph(torch.randn(3, 3, 32, 32))
        assert out.shape == (3, 100)


    def test_query_no_api(self):
        graph = NasBench201SearchSpace()
        graph.sample_random_architecture(dataset_api=create_dummy_api())

        try:
            results = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10')
        except Exception as e:
            assert isinstance(e, NotImplementedError)


    def test_query(self):
        graph = NasBench201SearchSpace()
        graph.set_spec((4, 3, 3, 0, 0, 0))

        api = create_dummy_api()
        api_data = api['nb201_data']['|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|']['cifar10-valid']

        val_acc = graph.query(Metric.VAL_ACCURACY, 'cifar10', dataset_api=api)
        assert val_acc == api_data['eval_acc1es'][-1]

        val_loss = graph.query(Metric.VAL_LOSS, 'cifar10', dataset_api=api)
        assert val_loss == api_data['eval_losses'][-1]

        train_acc = graph.query(Metric.TRAIN_ACCURACY, 'cifar10', dataset_api=api)
        assert train_acc == api_data['train_acc1es'][-1]

        train_loss = graph.query(Metric.TRAIN_LOSS, 'cifar10', dataset_api=api)
        assert train_loss == api_data['train_losses'][-1]

        val_acc_full = graph.query(Metric.VAL_ACCURACY, 'cifar10', dataset_api=api, full_lc=True)
        assert tuple(val_acc_full) == tuple(api_data['eval_acc1es'])

        val_acc_partial = graph.query(Metric.VAL_ACCURACY, 'cifar10', dataset_api=api, full_lc=True, epoch=50)
        assert tuple(val_acc_partial) == tuple(api_data['eval_acc1es'][:50])

        val_acc_50 = graph.query(Metric.VAL_ACCURACY, 'cifar10', dataset_api=api, full_lc=False, epoch=50)
        assert val_acc_50 == api_data['eval_acc1es'][50]

        hp = graph.query(Metric.HP, 'cifar10', dataset_api=api)
        assert hp == api_data['cost_info']

        train_time = graph.query(Metric.TRAIN_TIME, 'cifar10', dataset_api=api)
        assert train_time == api_data['cost_info']['train_time']


    def test_get_arch_iterator(self):
        graph = NasBench201SearchSpace()
        it = graph.get_arch_iterator()

        archs = list(it)

        assert len(archs) == 15625
        assert len(archs[0]) == 6

    def test_mutate(self):
        graph_parent = create_model()
        graph_child = NasBench201SearchSpace()

        graph_child.mutate(graph_parent)

        parent_spec = graph_parent.get_hash()
        child_spec = graph_child.get_hash()

        assert parent_spec != child_spec

        out = graph_child(torch.randn(3, 3, 32, 32))
        assert out.shape == (3, 10)

    def test_get_nbhd(self):
        graph = create_model()
        neighbours = graph.get_nbhd()

        print(len(neighbours))
        assert len(neighbours) == 24
