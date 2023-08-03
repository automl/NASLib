import logging
import torch

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph

from naslib.utils import count_parameters_in_MB

from fvcore.common.config import CfgNode

logger = logging.getLogger(__name__)


class LocalSearch(MetaOptimizer):
    """
    LocalSearch is a class for conducting local search in Neural Architecture Search (NAS) methods.
    It selects a random architecture, generates its neighborhood, and moves to a neighbor
    if it has better performance. If no better neighbors are found, a new random architecture is selected.

    Attributes:
        using_step_function (bool): Flag indicating the absence of a step function for this optimizer.
        config (CfgNode): Configuration settings for the search process.
        epochs (int): Number of epochs for the search process.
        performance_metric (Metric): The performance metric for evaluating the architectures.
        dataset (str): The dataset to be used for evaluation.
        num_init (int): Number of initial random architectures.
        nbhd (list): A list to store the neighborhood of the current architecture.
        chosen (Graph): The currently chosen architecture.
        best_arch (Graph): The best architecture found so far.
        history (torch.nn.ModuleList): A list to store the history of architectures.
        newest_child_idx (int): The index of the most recently added child architecture in the history.
    """

    # training the models is not implemented
    using_step_function = False

    def __init__(self, config: CfgNode):
        """
        Initializes the LocalSearch class with configuration settings.

        Args:
            config (CfgNode): Configuration settings for the search process.
        """
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.num_init = config.search.num_init
        self.nbhd = []
        self.chosen = None
        self.best_arch = None

        self.history = torch.nn.ModuleList()
        self.newest_child_idx = -1

    def adapt_search_space(self, search_space: Graph, scope: str = None, dataset_api: dict = None):
        """
        Adapts the search space for the local search.

        Args:
            search_space (Graph): The search space to be adapted.
            scope (str, optional): The scope for the search. Defaults to None.
            dataset_api (dict, optional): API for the dataset. Defaults to None.
        """
        assert (
            search_space.QUERYABLE
        ), "Local search is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, epoch: int):
        """
        Starts a new epoch in the search process, performing local search on the chosen architecture.

        Args:
            epoch (int): The current epoch number.
        """

        if epoch < self.num_init:
            # randomly sample initial architectures
            model = (
                torch.nn.Module()
            )
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = model.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

            if not self.best_arch or model.accuracy > self.best_arch.accuracy:
                self.best_arch = model
            self._update_history(model)

        else:
            if (
                len(self.nbhd) == 0
                and self.chosen
                and self.best_arch.accuracy <= self.chosen.accuracy
            ):
                logger.info(
                    "Reached local minimum. Starting from new random architecture."
                )

                model = (
                    torch.nn.Module()
                )
                model.arch = self.search_space.clone()
                model.arch.sample_random_architecture(dataset_api=self.dataset_api)
                model.accuracy = model.arch.query(
                    self.performance_metric, self.dataset, dataset_api=self.dataset_api
                )

                self.chosen = model
                self.best_arch = model
                self.nbhd = self.chosen.arch.get_nbhd(dataset_api=self.dataset_api)

            else:
                if len(self.nbhd) == 0:
                    logger.info(
                        "Start a new iteration. Pick the best architecture and evaluate its neighbors."
                    )
                    self.chosen = self.best_arch
                    self.nbhd = self.chosen.arch.get_nbhd(dataset_api=self.dataset_api)

                model = self.nbhd.pop()
                model.accuracy = model.arch.query(
                    self.performance_metric, self.dataset, dataset_api=self.dataset_api
                )

                if model.accuracy > self.best_arch.accuracy:
                    self.best_arch = model
                    logger.info("Found new best architecture.")
                self._update_history(model)

    def _update_history(self, child):
        """
        Updates the history of architectures with a new child architecture.

        Args:
            child (torch.nn.Module): The new child architecture to be added to the history.
        """
        if len(self.history) < 100:
            self.history.append(child)
            self.newest_child_idx = len(self.history) - 1
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    self.newest_child_idx = i
                    break

    def train_statistics(self, report_incumbent: bool = True):
        """
        Reports the statistics after training.

        Args:
            report_incumbent (bool, optional): Whether to report the incumbent or the most recent architecture. Defaults to True.

        Returns:
            tuple: A tuple containing the training accuracy, validation accuracy, and test accuracy, and training time.
        """
        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.history[self.newest_child_idx].arch
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

    def get_final_architecture(self):
        """
        Gets the final (best) architecture from the search.

        Returns:
            Graph: The best architecture found during the search.
        """
        return max(self.history, key=lambda x: x.accuracy).arch

    def get_op_optimizer(self):
        """
        Gets the optimizer for the operations. This method is not implemented in this class and raises an error when called.

        Raises:
            NotImplementedError: Always, because this method is not implemented in this class.
        """
        raise NotImplementedError()

    def get_checkpointables(self):
        """
        Gets the models that can be checkpointed.

        Returns:
            dict: A dictionary with "model" as the key and the history of architectures as the value.
        """
        return {"model": self.history}

    def get_model_size(self):
        """
        Gets the size of the model.

        Returns:
            float: The size of the model in megabytes (MB).
        """
        return count_parameters_in_MB(self.history)
