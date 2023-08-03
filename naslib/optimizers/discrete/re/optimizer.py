import collections
import logging
import torch
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph

from naslib.utils import count_parameters_in_MB
from naslib.utils.log import log_every_n_seconds

from fvcore.common.config import CfgNode

logger = logging.getLogger(__name__)


class RegularizedEvolution(MetaOptimizer):
    """
    RegularizedEvolution is a class that implements the Regularized Evolution algorithm for
    Neural Architecture Search (NAS).

    Attributes:
        using_step_function (bool): Flag indicating the absence of a step function for this optimizer.
        config (CfgNode): Configuration node with settings for the search process.
        epochs (int): Number of epochs for the search process.
        sample_size (int): The number of architectures to sample for each population.
        population_size (int): The maximum size of the population in the evolutionary search.
        performance_metric (Metric): The performance metric for evaluating the architectures.
        dataset (str): The dataset to be used for evaluation.
        population (collections.deque): A queue to hold the population of architectures.
        history (torch.nn.ModuleList): A list to store the history of architectures.
    """
    using_step_function = False

    def __init__(self, config: CfgNode):
        """
        Initializes the Regularized Evolution class with configuration settings.

        Args:
            config (CfgNode): Configuration settings for the search process.
        """
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs
        self.sample_size = config.search.sample_size
        self.population_size = config.search.population_size

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.population = collections.deque(maxlen=self.population_size)
        self.history = torch.nn.ModuleList()

    def adapt_search_space(self, search_space: Graph, scope: str = None, dataset_api: dict = None, **kwargs):
        """
        Adapts the search space for regularized evolution search.

        Args:
            search_space (Graph): The search space to be adapted.
            scope (str, optional): The scope for the search. Defaults to None.
            dataset_api (dict, optional): API for the dataset. Defaults to None.
        """
        assert (
            search_space.QUERYABLE
        ), "Regularized evolution is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, epoch: int):
        """
        Starts a new epoch in the search process.

        Args:
            epoch (int): The current epoch number.
        """
        # We sample as many architectures as we need
        if epoch < self.population_size:
            logger.info("Start sampling architectures to fill the population")
            # If there is no scope defined, let's use the search space default one

            model = (
                torch.nn.Module()
            )
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = model.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

            self.population.append(model)
            self._update_history(model)
            log_every_n_seconds(
                logging.INFO, "Population size {}".format(len(self.population))
            )
        else:
            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)

            parent = max(sample, key=lambda x: x.accuracy)

            child = (
                torch.nn.Module()
            )
            child.arch = self.search_space.clone()
            child.arch.mutate(parent.arch, dataset_api=self.dataset_api)
            child.accuracy = child.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

            self.population.append(child)
            self._update_history(child)

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
            best_arch = self.population[-1].arch

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
        Gets the size of the model in terms of the number of parameters.

        Returns:
            float: The size of the model in MB.
        """
        return count_parameters_in_MB(self.history)
