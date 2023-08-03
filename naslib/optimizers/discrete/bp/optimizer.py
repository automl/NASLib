import logging
import torch
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.predictors.ensemble import Ensemble

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph

from naslib.utils import count_parameters_in_MB

from fvcore.common.config import CfgNode

logger = logging.getLogger(__name__)


class BasePredictor(MetaOptimizer):
    """
    BasePredictor is a base class for prediction-based Neural Architecture Search (NAS) methods.
    It provides the foundation for methods that use machine learning models to predict the performance of
    neural architectures without having to train them. Derived from the MetaOptimizer class.

    Attributes:
        using_step_function (bool): Flag indicating the absence of a step function for this optimizer.
        config (CfgNode): Configuration settings for the search process.
        epochs (int): Number of epochs for the search process.
        performance_metric (Metric): The performance metric for evaluating the architectures.
        dataset (str): The dataset to be used for evaluation.
        k (int): Number of architectures to be evaluated in each cycle.
        num_init (int): Number of initial random architectures.
        test_size (int): Size of the test set for evaluating the predictor.
        predictor_type (str): Type of predictor to use (e.g., "LGB", "MLP").
        num_ensemble (int): Number of models in the ensemble.
        encoding_type (str): Type of encoding for the architectures (e.g., "adjacency_one_hot").
        debug_predictor (bool): If True, debug information will be printed.
        train_data (list): A list to store the training data (architecture-performance pairs).
        choices (list): A list to store the chosen architectures.
        history (torch.nn.ModuleList): A list to store the history of architectures.
    """
    # training the models is not implemented
    using_step_function = False

    def __init__(self, config: CfgNode):
        """
        Initializes the BasePredictor class with configuration settings.

        Args:
            config (CfgNode): Configuration settings for the search process.
        """
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        # 20, 172 are the magic numbers from [Wen et al. 2019]
        self.k = int(20 / 172 * self.epochs)
        self.num_init = self.epochs - self.k
        self.test_size = 2 * self.epochs

        self.predictor_type = config.search.predictor_type
        self.num_ensemble = config.search.num_ensemble
        self.encoding_type = config.search.encoding_type
        self.debug_predictor = config.search.debug_predictor

        self.train_data = []
        self.choices = []
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
        ), "Regularized evolution is currently only implemented for benchmarks."

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, epoch: int):
        """
        Starts a new epoch in the search process, sampling a new architecture to train or using the predictor
        to select an architecture.

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

            self.train_data.append(model)
            self._update_history(model)

        else:
            if epoch == self.num_init:
                # train the neural predictor and use it to predict arches in test_data

                xtrain = [m.arch for m in self.train_data]
                ytrain = [m.accuracy for m in self.train_data]

                ensemble = Ensemble(
                    encoding_type=self.encoding_type,
                    num_ensemble=self.num_ensemble,
                    predictor_type=self.predictor_type,
                    ss_type=self.search_space.get_type(),
                )
                # train_error = ensemble.fit(xtrain, ytrain)
                ensemble.fit(xtrain, ytrain)

                xtest = []
                for i in range(self.test_size):
                    arch = self.search_space.clone()
                    arch.sample_random_architecture(dataset_api=self.dataset_api)
                    xtest.append(arch)

                test_pred = np.squeeze(ensemble.query(xtest))
                test_pred = np.mean(test_pred, axis=0)

                if self.debug_predictor:
                    self.evaluate_predictor(
                        xtrain=xtrain, ytrain=ytrain, xtest=xtest, test_pred=test_pred
                    )

                sorted_indices = np.argsort(test_pred)[-self.k:]
                for i in sorted_indices:
                    self.choices.append(xtest[i])

            # train the next chosen architecture
            choice = (
                torch.nn.Module()
            )
            choice.arch = self.choices[epoch - self.num_init]
            choice.accuracy = choice.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )
            self._update_history(choice)

    def evaluate_predictor(self, xtrain, ytrain, xtest, test_pred, slice_size: int = 4):
        """
        Evaluates the predictor for debugging purposes.

        Args:
            xtrain: Training data (architectures).
            ytrain: Training labels (performances).
            xtest: Test data (architectures).
            test_pred: Predicted performances for the test data.
            slice_size (int, optional): Number of items to print in each slice. Defaults to 4.
        """
        ytest = []
        for arch in xtest:
            ytest.append(
                arch.query(
                    self.performance_metric, self.dataset, dataset_api=self.dataset_api
                )
            )

        print("ytrain shape", np.array(ytrain).shape)
        print("ytest shape", np.array(ytest).shape)
        print("test_pred shape", np.array(test_pred).shape)
        test_error = np.mean(abs(test_pred - ytest))
        correlation = np.corrcoef(np.array(ytest), np.array(test_pred))[1, 0]
        print("test error", test_error)
        print("correlation", correlation)
        print()
        print("xtrain slice", xtrain[:slice_size])
        print("ytrain slice", ytrain[:slice_size])
        print()
        print("xtest slice", xtest[:slice_size])
        print("ytest slice", ytest[:slice_size])
        print("test_pred slice", test_pred[:slice_size])

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
            tuple: A tuple containing the training accuracy, validation accuracy, and test accuracy.
        """
        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.history[-1].arch
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
