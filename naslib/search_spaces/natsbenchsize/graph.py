import numpy as np
import random
import itertools
import torch

from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric


class NATSBenchSizeSearchSpace(Graph):
    """
    Implementation of the nasbench 201 search space.
    It also has an interface to the tabular benchmark of nasbench 201.
    """

    QUERYABLE = True

    def __init__(self):
        super().__init__()
        self.channel_candidates = [8*i for i in range(1, 9)]
        self.channels = [8, 8, 8, 8, 8]

        self.space_name = "natsbenchsizesearchspace"
        # Graph not implemented

    def query(
        self,
        metric=None,
        dataset=None,
        path=None,
        epoch=-1,
        full_lc=False,
        dataset_api=None,
        hp=90,
        is_random=False
    ):
        """
        Query results from natsbench

        Args:
            metric      : Metric to query for
            dataset     : Dataset to query for
            epoch       : If specified, returns the metric of the arch at that epoch of training
            full_lc     : If true, returns the curve of the given metric from the first to the last epoch
            dataset_api : API to use for querying metrics
            hp          : Number of epochs the model was trained for. Value is in {1, 12, 90}
            is_random   : When True, the performance of a random architecture will be returned
                          When False, the performanceo of all trials will be averaged.
        """
        assert isinstance(metric, Metric)
        assert dataset in [
            "cifar10",
            "cifar100",
            "ImageNet16-120",
        ], "Unknown dataset: {}".format(dataset)
        assert epoch >= -1 and epoch < hp
        assert hp in [1, 12, 90], "hp must be 1, 12 or 90"
        if dataset=='cifar10':
            assert metric not in [Metric.VAL_ACCURACY, Metric.VAL_LOSS, Metric.VAL_TIME],\
            "Validation metrics not available for CIFAR-10"

        metric_to_natsbench = {
            Metric.TRAIN_ACCURACY: "train-accuracy",
            Metric.VAL_ACCURACY: "valid-accuracy",
            Metric.TEST_ACCURACY: "test-accuracy",
            Metric.TRAIN_LOSS: "train-loss",
            Metric.VAL_LOSS: "valid-loss",
            Metric.TEST_LOSS: "test-loss",
            Metric.TRAIN_TIME: "train-all-time",
            Metric.VAL_TIME: "valid-all-time",
            Metric.TEST_TIME: "test-all-time"
        }

        if metric not in metric_to_natsbench.keys():
            raise NotImplementedError(f"NATS-Bench does not support querying {metric}")
        if dataset_api is None:
            raise NotImplementedError("Must pass in dataset_api to query natsbench")

        arch_index = int(''.join([str(ch//8 - 1) for ch in self.channels]), 8)

        if epoch == -1:
            epoch = hp - 1
        hp = f"{hp:02d}"

        if full_lc:
            metrics = []

            for epoch in range(int(hp)):
                result = dataset_api.get_more_info(arch_index, dataset, iepoch=epoch, hp=hp, is_random=is_random)
                metrics.append(result[metric_to_natsbench[metric]])

            return metrics
        else:
            results = dataset_api.get_more_info(arch_index, dataset, iepoch=epoch, hp=hp, is_random=is_random)
            return results[metric_to_natsbench[metric]]

    def get_channels(self):
        return self.channels

    def set_channels(self, channels):
        self.channels = channels

    def get_hash(self):
        return tuple(self.get_channels())

    def get_arch_iterator(self, dataset_api=None):
        return itertools.product(self.channel_candidates, repeat=len(self.channels))

    def set_spec(self, channels, dataset_api=None):
        # this is just to unify the setters across search spaces
        # TODO: change it to set_spec on all search spaces
        self.set_channels(channels)

    def sample_random_architecture(self, dataset_api=None):
        """
        Randomly sample an architecture
        """
        channels = np.random.choice(self.channel_candidates, size=len(self.channels)).tolist()
        self.set_channels(channels)

    def mutate(self, parent, dataset_api=None):
        """
        Mutate one channel from the parent channels
        """

        base_channels = list(parent.get_channels().copy())
        mutate_index = np.random.randint(len(self.channels)) # Index to perform mutation at

        # Remove number of channels at that index in base_channels from the viable candidates
        candidates = self.channel_candidates.copy()
        candidates.remove(base_channels[mutate_index])

        base_channels[mutate_index] = np.random.choice(candidates)
        self.set_channels(base_channels)

    def get_nbhd(self, dataset_api=None):
        """
        Return all neighbours of the architecture
        """
        neighbours = []

        for idx in range(len(self.channels)):
            candidates = self.channel_candidates.copy()
            candidates.remove(self.channels[idx])

            for channels in candidates:
                neighbour_channels = list(self.channels).copy()
                neighbour_channels[idx] = channels
                neighbour = NATSBenchSizeSearchSpace()
                neighbour.set_channels(neighbour_channels)
                neighbour_model = torch.nn.Module()
                neighbour_model.arch = neighbour
                neighbours.append(neighbour_model)

        random.shuffle(neighbours)
        return neighbours

    def get_type(self):
        return "natsbenchsize"

