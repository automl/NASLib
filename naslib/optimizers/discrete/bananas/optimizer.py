import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.utils.utils import sample_random_architecture, mutate, get_op_indices
from naslib.optimizers.discrete.utils.encodings import encode
from naslib.optimizers.discrete.predictor.predictors import Ensemble
from naslib.optimizers.discrete.bananas.acquisition_functions import acquisition_function



from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds



logger = logging.getLogger(__name__)


class Bananas(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.k = 10
        self.num_init = 10
        self.num_ensemble = 3
        self.acq_fn_type = 'its'
        self.acq_fn_optimization = 'mutation'
        self.encoding_type = 'path'
        self.num_arches_to_mutate = 2
        self.max_mutations = 1
        self.num_candidates = 100

        self.train_data = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()



    def adapt_search_space(self, search_space, scope=None):
        assert search_space.QUERYABLE, "Bananas is currently only implemented for benchmarks."
        
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE        
    
    
    def new_epoch(self, epoch):

        if epoch < self.num_init:
            logger.info("Start sampling architectures to generate training data")
            # If there is no scope defined, let's use the search space default one
            
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            
            model.arch = sample_random_architecture(self.search_space, self.scope)
            model.accuracy = model.arch.query(self.performance_metric, self.dataset)
            
            self.train_data.append(model)
            self._update_history(model)

        else:
            if len(self.next_batch) == 0:
                # train a neural predictor
                xtrain = [encode(m.arch, encoding_type=self.encoding_type) for m in self.train_data]
                ytrain = [m.accuracy for m in self.train_data]
                ensemble = Ensemble(self.num_ensemble)
                train_error = ensemble.fit(xtrain, ytrain)

                # define an acquisition function
                acq_fn = acquisition_function(ensemble=ensemble, 
                                              ytrain=ytrain,
                                              acq_fn_type=self.acq_fn_type)
                
                # optimize the acquisition function to output k new architectures
                candidates = []
                if self.acq_fn_optimization == 'random_sampling':
                    
                    for _ in range(self.num_candidates):
                        arch = sample_random_architecture(self.search_space, self.scope)
                        candidates.append(arch)
                    
                elif self.acq_fn_optimization == 'mutation':
                    # mutate the k best architectures by x
                    best_arch_indices = np.argsort(ytrain)[-self.num_arches_to_mutate:]
                    best_arches = [self.train_data[i].arch for i in best_arch_indices]
                    candidates = []
                    for arch in best_arches:
                        for _ in range(int(self.num_candidates / len(best_arches) / self.max_mutations)):
                            candidate = arch.clone()
                            for edit in range(int(self.max_mutations)):
                                candidate = mutate(candidate)
                            candidates.append(candidate)

                else:
                    logging.info('{} is not yet supported as a acq fn optimizer'.format(encoding_type))
                    raise NotImplementedError()

                candidate_encodings = [encode(arch, encoding_type=self.encoding_type) for arch in candidates]
                values = [acq_fn(encoding) for encoding in candidate_encodings]
                sorted_indices = np.argsort(values)
                choices = [candidates[i] for i in sorted_indices[-self.k:]]                        
                self.next_batch = [*choices]

            # train the next architecture chosen by the neural predictor
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable            
            model.arch = self.next_batch.pop()
            model.accuracy = model.arch.query(self.performance_metric, self.dataset)
            self._update_history(model)
            self.train_data.append(model)
        
    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self):
        best_arch = self.get_final_architecture()
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset), 
            best_arch.query(Metric.TRAIN_LOSS, self.dataset), 
            best_arch.query(Metric.VAL_ACCURACY, self.dataset), 
            best_arch.query(Metric.VAL_LOSS, self.dataset), 
        )
    

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset)


    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch
    
    
    def get_op_optimizer(self):
        raise NotImplementedError()

    
    def get_checkpointables(self):
        return {'model': self.history}
    

    def get_model_size(self):
        return count_parameters_in_MB(self.history)