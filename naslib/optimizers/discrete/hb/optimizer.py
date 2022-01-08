import numpy as np
import torch
import copy
import math
from collections import defaultdict
from naslib.optimizers import SuccessiveHalving as SH
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.predictors import ensemble
from naslib.predictors.ensemble import Ensemble
from naslib.search_spaces.core.query_metrics import Metric

from collections import defaultdict


class HyperBand(MetaOptimizer):
    """
    This is a Hyperband Implementation, that uses the Sucessive Halving Algorithm with different settings.
    """

    # training the models is not implemented
    using_step_function = False

    def __init__(
        self,
        config,
        weight_optimizer=torch.optim.SGD,
        loss_criteria=torch.nn.CrossEntropyLoss(),
        grad_clip=None,
    ):
        """
        Initialize a Successive Halving  optimizer.

        Args:
            config
            weight_optimizer (torch.optim.Optimizer): The optimizer to
                train the (convolutional) weights.
            loss_criteria (TODO): The loss
            grad_clip (float): Where to clip the gradients (default None).
        """
        super(HyperBand, self).__init__()
        self.weight_optimizer = weight_optimizer
        self.loss = loss_criteria
        self.grad_clip = grad_clip
        self.sh_config = copy.deepcopy(config)
        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset
        self.method = config.search.method
        #self.fidelit_min = config.search.min_fidelity
        self.budget_max = config.search.budget_max
        self.eta = config.search.eta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.budget_type = config.search.budget_type #is not for one query is overall
        self.sampled_archs = []
        self.history = torch.nn.ModuleList()
        self.s_max = math.floor(math.log(self.budget_max, self.eta))
        self.s = self.s_max + 1
        self.sh = None
        self.first = True # TODO: think about a more ellegant solution 
        self.b = (self.s_max + 1)* self.budget_max
        self.encoding_type = config.search.encoding_type
        self.ss_type= config.search_space
        self.optimizer_stats = defaultdict(lambda: defaultdict(list))
        
    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Successsive Halving is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        if self.method == "tpe":
            
            self.ensemble =   Ensemble(
                    predictor_type= "tpe",  
                    num_ensemble= 1,
                    encoding_type= self.encoding_type,
                    ss_type=  self.ss_type,
                    config = [self.search_space.clone(),self.dataset_api, self.dataset],   #replace with config maybe or not 

                    )

    def new_epoch(self):
        """
        Sample a new architecture to train.
        after https://arxiv.org/pdf/1603.06560.pdf
        """
        # TODO: rethink because of s
        if  self.sh == None or self.sh.end:
            #if sh is finish go to something diffrent as initial budget
                    #n = ((self.b ) / (self.budget_max))* ((self.eta**self.s)/(self.s + 1))
            self.s -= 1
            if self.sh != None:
                self.sampled_archs.extend(self.sh.sampled_archs)
            if self.s < 0:
                print("HB is finish, allready not defined what to do") # TODO define what to do 
                return math.inf #end the thing
            n = math.ceil(int(self.b / self.budget_max / (self.s + 1)) * self.eta ** self.s)
            r = self.budget_max * self.eta ** (-self.s)
            self.sh_config.search.number_archs = n
            self.sh_config.search.min_fidelity  = r 

            self.sh = SH(self.sh_config,  esemble = self.ensemble) #should be in config 
            self.sh.adapt_search_space(self.search_space, dataset_api= self.dataset_api)
        
        budget = self.sh.new_epoch()

        self.update_optimizer_stats()
        return budget
                
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
        all_sampled = copy.deepcopy(self.sampled_archs) #This is because we also want the running into accound, but not later
        all_sampled.extend(self.sh.sampled_archs)
        return max(all_sampled, key=lambda x: x.accuracy).arch

    def train_statistics(self, report_incumbent=True):
        
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
    def train_model_statistics(self, report_incumbent=True):

        
        best_arch = self.sampled_archs[self.fidelity_counter -1].arch
        best_arch_hash = hash(self.sampled_archs[self.fidelity_counter -1])
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
            self.fidelity,
            best_arch_hash,
        )
    
    def update_optimizer_stats(self):
        """
        Updates statistics of optimizer to be able to create useful plots
        TODO i have to expand the dictionary such that we keep track of all parallel sh evaluations
        """
        self.optimizer_stats[self.s] = self.sh.optimizer_stats

    def update_optimizer_stats(self):
        """
        Updates statistics of optimizer to be able to create useful plots
        here only hacky way from sh
        """
        arch = self.sh.sampled_archs[self.sh.fidelity_counter-1].arch
        arch_hash = hash(self.sh.sampled_archs[self.sh.fidelity_counter- 1])
        # this dict contains metrics to save
        metrics = {
            "train_acc": Metric.TRAIN_ACCURACY,
            "val_acc": Metric.VAL_ACCURACY,
            "test_acc": Metric.TEST_ACCURACY,
            "train_time": Metric.TRAIN_TIME
        }
        for metric_name, metric in metrics.items():
            metric_value = arch.query(
                metric, 
                self.dataset, 
                dataset_api=self.dataset_api, 
                epoch=int(81)
            )
            self.optimizer_stats[arch_hash][metric_name].append(metric_value)
        self.optimizer_stats[arch_hash]['fidelity'].append(self.sh.fidelity) 

    def test_statistics(self):
        return False
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_op_optimizer(self):
        return self.weight_optimizer

    def get_checkpointables(self):
        return {"model": self.history}
