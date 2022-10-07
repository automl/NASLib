import torch

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph

from fvcore.common.config import CfgNode

class RandomSearch(MetaOptimizer):
    """
    Random search in DARTS is done by randomly sampling `k` architectures
    and training them for `n` epochs, then selecting the best architecture.
    DARTS paper: `k=24` and `n=100` for cifar-10.
    """

    using_step_function = False

    def __init__(
            self,
            config: CfgNode
    ):
        """
        Initialize a random search optimizer.

        Args:
            config: Config file
        """
        super(RandomSearch, self).__init__()

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.fidelity = config.search.fidelity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sampled_archs = []
        self.history = torch.nn.ModuleList()

    def adapt_search_space(self, search_space: Graph, scope: str = None, dataset_api: dict = None):
        assert (
            search_space.QUERYABLE
        ), "Random search is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, epoch: int):
        """
        Sample a new architecture to train.
        """

        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(dataset_api=self.dataset_api)
        model.accuracy = model.arch.query(
            self.performance_metric,
            self.dataset,
            epoch=self.fidelity,
            dataset_api=self.dataset_api,
        )

        self.sampled_archs.append(model)
        self._update_history(model)

    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def get_final_architecture(self):
        """
        Returns the sampled architecture with the lowest validation error.
        """
        return max(self.sampled_archs, key=lambda x: x.accuracy).arch

    def train_statistics(self, report_incumbent: bool = True):

        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.sampled_archs[-1].arch

        return (
            best_arch.query(
                Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
            ),
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_op_optimizer(self):
        raise NotImplementedError

    def get_checkpointables(self):
        return {"model": self.history}
