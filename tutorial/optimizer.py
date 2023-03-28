import torch

from fvcore.common.config import CfgNode

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces.core.graph import Graph
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_train_val_loaders

# Import whatever else you want from NASLib

class AwesomeOptimizer(MetaOptimizer):
    """
    Implement your own awesome optimizer here.

    This optimizer inherits from RandomSearch purely for convenience. Your search
    method does not have to be random at all. Feel free to write your own logic, and add 
    any new methods that you need for it.
    """

    def __init__(self, config: CfgNode):
        """
        Initialize your optimizer

        Args:
            config
        """
        super(AwesomeOptimizer, self).__init__()
        # These two lists are required
        self.sampled_archs = []
        self.history = []
        
        self.performance_metric = Metric.VAL_ACCURACY
        # You can add more properties to the config file and read them if you want
        self.dataset = config.dataset
        self.fidelity = config.search.fidelity

        # A few things that might be useful to you. Add/remove code as you wish.
        self.train_dataloader = get_train_val_loaders(config)[0]
        self.zerocostpredictor = ZeroCost(method_type='l2_norm')

        ###########################################################
        ##################### START TODO ##########################


        # Need more stuff in your initializer? Write them here!


        ##################### END TODO  ##########################
        ##########################################################


    def adapt_search_space(self, search_space: Graph, scope: str = None, dataset_api: dict = None):
        """
        This method has to be called with the search_space and the nas benchmark api before the optimizer
        can be used.

        Args:
            search_space : An instance of the search space, such as NasBench201SearchSpace()
            scope        : Relevant only for one-shot optimizers
            dataset_api  : NAS Benchmark API for the given search space
        """
        self.search_space = search_space.clone()
        self.dataset_api = dataset_api


    def new_epoch(self, epoch: int):
        """
        This method is called in every "step" of the search.

        Args:
            epoch: epoch number
        """

        # For reference, this is the code inside new_epoch in RandomSearch:

        # model = torch.nn.Module()
        # model.arch = self.search_space.clone()
        # model.arch.sample_random_architecture(dataset_api=self.dataset_api)
        # model.accuracy = model.arch.query(
        #     self.performance_metric,
        #     self.dataset,
        #     epoch=self.fidelity,
        #     dataset_api=self.dataset_api,
        # )
        # self.sampled_archs.append(model)
        # self._update_history(model)

        # All it does 

        ###########################################################
        ##################### START TODO ##########################


        # Write your logic here
        # Also feel free to write new methods in this class


        ##################### END TODO  ##########################
        ##########################################################

        self.sampled_archs.append(model) # This line is required. Add your chosen model to sampled_archs here.

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

    def _update_history(self, child):
        """
            We want to maintain only the history of the 100 best models in self.history
            (self.sampled_archs stores the list of all models sampled.)
        """
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def get_checkpointables(self):
        return {"models": self.history}
