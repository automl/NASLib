import torch

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph

from fvcore.common.config import CfgNode


class RandomSearch(MetaOptimizer):
    """
    RandomSearch is a class that implements the Random Search algorithm for
    Neural Architecture Search (NAS). It is derived from the MetaOptimizer class.
    The random search is done by randomly sampling 'k' architectures and training
    them for 'n' epochs, then selecting the best architecture.
    In the DARTS paper, 'k' equals 24 and 'n' equals 100 for the CIFAR-10 dataset.

    Attributes:
        using_step_function (bool): Flag indicating the absence of a step function for this optimizer.
        performance_metric (Metric): The performance metric for evaluating the architectures.
        dataset (str): The dataset to be used for evaluation.
        fidelity (int): The number of epochs for each sampled architecture's training.
        device (torch.device): The device to be used for computations, either CUDA or CPU.
        sampled_archs (list): A list to store the sampled architectures.
        history (torch.nn.ModuleList): A list to store the history of architectures.
    """

    using_step_function = False

    def __init__(
            self,
            config: CfgNode
    ):
        """
        Initializes the RandomSearch class with configuration settings.

        Args:
            config (CfgNode): Configuration settings for the search process.
        """
        super(RandomSearch, self).__init__()

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.fidelity = config.search.fidelity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sampled_archs = []
        self.history = torch.nn.ModuleList()

    def adapt_search_space(self, search_space: Graph, scope: str = None, dataset_api: dict = None):
        """
        Adapts the search space for the random search.

        Args:
            search_space (Graph): The search space to be adapted.
            scope (str, optional): The scope for the search. Defaults to None.
            dataset_api (dict, optional): API for the dataset. Defaults to None.
        """
        assert (
            search_space.QUERYABLE
        ), "Random search is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, epoch: int):
        """
        Starts a new epoch in the search process, sampling a new architecture to train.

        Args:
            epoch (int): The current epoch number.
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
        """
        Updates the history of architectures with a new child architecture.

        Args:
            child (torch.nn.Module): The new child architecture to be added to the history.
        """
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def get_final_architecture(self):
        """
        Gets the final (best) architecture from the search.

        Returns:
            Graph: The best architecture found during the search.
        """
        return max(self.sampled_archs, key=lambda x: x.accuracy).arch

    def train_statistics(self, report_incumbent: bool = True):
        """
        Reports the statistics after training.

        Args:
            report_incumbent (bool, optional): Whether to report the incumbent or the most recent architecture. Defaults to True.

        Returns:
            tuple: A tuple containing the training accuracy, validation accuracy, test accuracy, and training time.
        """
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
        """
        Reports the test statistics.

        Returns:
            float: The raw performance metric for the best architecture.
        """
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_op_optimizer(self):
        """
        Gets the optimizer for the operations. This method is not implemented in this class and raises an error when called.

        Raises:
            NotImplementedError: Always, because this method is not implemented in this class.
        """
        raise NotImplementedError

    def get_checkpointables(self):
        """
        Gets the models that can be checkpointed.

        Returns:
            dict: A dictionary with "model" as the key and the history of architectures as the value.
        """
        return {"model": self.history}
