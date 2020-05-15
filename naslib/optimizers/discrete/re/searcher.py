import collections
import logging
from copy import deepcopy

import numpy as np

from naslib.optimizers.discrete import Searcher as BaseSearcher
from naslib.utils.utils import AttrDict


class Searcher(BaseSearcher):
    def __init__(self, graph, parser, arch_optimizer, *args, **kwargs):
        super(Searcher, self).__init__(graph, parser, arch_optimizer, *args, **kwargs)

        self.population = collections.deque()
        self.history = []
        self.population_size = parser.config.population_size
        self.sample_size = parser.config.sample_size

    def run(self, n_evaluations, *args, **kwargs):
        while len(self.population) < self.population_size:
            model = AttrDict()
            # sample a random arch
            self.arch_optimizer.uniform_sample()
            model.arch = deepcopy(self.arch_optimizer.architectural_weights)
            arch_info = self.query()
            model.accuracy = arch_info['cifar10-valid']['valid_accuracy']
            self.population.append(model)
            self.history.append(model)

        while len(self.history) < n_evaluations:
            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)

            parent = max(sample, key=lambda x: x.accuracy)

            child = AttrDict()
            self.arch_optimizer.mutate_arch(deepcopy(parent.arch))
            child.arch = deepcopy(self.arch_optimizer.architectural_weights)
            arch_info = self.query()
            child.accuracy = arch_info['cifar10-valid']['valid_accuracy']
            self.population.append(child)
            self.history.append(child)

            self.population.popleft()

    # NOTE: this works only for nasbench201 for now
    def query(self):
        if hasattr(self.graph, 'query_architecture'):
            # Record anytime performance
            arch_info = self.graph.query_architecture(self.arch_optimizer.architectural_weights)
            logging.info('arch {}'.format(arch_info))
            if 'arch_eval' not in self.errors_dict:
                self.errors_dict['arch_eval'] = []
            self.errors_dict['arch_eval'].append(arch_info)
            self.log_to_json(self.parser.config.save)
            return arch_info
