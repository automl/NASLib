import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.rs.optimizer import sample_random_architecture

from naslib.utils.utils import AttrDict
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)

class Counter():

    def __init__(self, i=0):
        self.i = i

    def increment(self):
        self.i += 1

    def decrement(self):
        self.i -= 1


class RegularizedEvolution(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.epochs
        self.sample_size = config.sample_size
        self.population_size = config.population_size

        self.performance_metric = 'eval_acc1es'

        self.population = collections.deque(maxlen=self.population_size)
        self.history = []


    def adapt_search_space(self, search_space, scope=None):

        assert search_space.QUERYABLE
        
        # We sample as many architectures as we need
        logger.info("Start sampling architectures to fill the population")
        while len(self.population) < self.population_size:
            # If there is no scope defined, let's use the search space default one
            if not scope:
                scope = search_space.OPTIMIZER_SCOPE
            
            model = AttrDict()
            
            model.arch = sample_random_architecture(search_space, scope)
            model.accuracy = model.arch.query(self.performance_metric)
            
            self.population.append(model)
            self.history.append(model)
            log_every_n_seconds(logging.INFO, "Population size {}".format(len(self.population)))

    @staticmethod
    def mutate_op(current_edge_data, mut_counter, seen_counter):
        seen_counter.decrement()    # keep track to mutate at lest the last one
        num_mutations = 1
        if mut_counter.i < num_mutations:
            if np.random.randint(len(current_edge_data.primitives)) == 0:
                op_index = np.random.randint(len(current_edge_data.primitives))
                current_edge_data.set('op_index', op_index, shared=True)
                mut_counter.increment()
            elif seen_counter.i == 0:
                op_index = np.random.randint(len(current_edge_data.primitives))
                current_edge_data.set('op_index', op_index, shared=True)
                mut_counter.increment()
        return current_edge_data



    def _mutate(self, parent_arch):
        child = parent_arch.clone()
        
        # sample which cell/motif we want to mutate
        cells = child._get_child_graphs(single_instances=True)
        cell = np.random.choice(cells) if len(cells) > 1 else cells[0]
        
        # sample if op or edge change
        if True: #np.random.choice(a=[False, True]):
            # change op
            mut_c = Counter()    # keep track of the number of mutations
            seen_c = Counter(len(cell.get_all_edge_data('op_index')))    # keep track of the number of edges we have seen
            cell.update_edges(
                lambda current_edge_data: self.mutate_op(current_edge_data, mut_c, seen_c), 
                private_edge_data=False
            )
        else:
            # change edge
            pass
        return child
    
    
    def new_epoch(self, epoch):
        sample = []
        while len(sample) < self.sample_size:
            candidate = np.random.choice(list(self.population))
            sample.append(candidate)
        
        parent = max(sample, key=lambda x: x.accuracy)

        child = AttrDict()
        child.arch = self._mutate(parent.arch)
        child.accuracy = child['arch'].query(self.performance_metric)

        self.population.append(child)
        self.history.append(child)
        

    def train_statistics(self):
        best_arch = max(self.population, key=lambda x: x.accuracy)['arch']
        return best_arch.query('train_acc1es'), best_arch.query('train_losses'), best_arch.query('eval_acc1es'), best_arch.query('eval_losses'), 
    
    def test_statistics(self):
        return 0, 0

    def get_final_architecture(self):
        return max(self.population, key=lambda x: x.accuracy)['arch']
    
    def get_op_optimizer(self):
        raise NotImplementedError()

