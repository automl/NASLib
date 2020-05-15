import logging
from copy import deepcopy
from hyperopt import fmin, tpe, STATUS_OK, Trials

from naslib.optimizers.discrete import Searcher as BaseSearcher
from naslib.utils.utils import AttrDict


class Searcher(BaseSearcher):
    def __init__(self, graph, parser, arch_optimizer, *args, **kwargs):
        super(Searcher, self).__init__(graph, parser, arch_optimizer, *args, **kwargs)

        self.trials = Trials()

    def run(self, n_evaluations, *args, **kwargs):
        best = fmin(self.objective, space=self.arch_optimizer.space,
                    algo=tpe.suggest, max_evals=n_evaluations,
                    trials=self.trials)

    def objective(self, x):
        config = deepcopy(x)
        print('CONFIG: ', config)
        self.arch_optimizer.set_to_zero()
        for arch_key, arch_weight in self.arch_optimizer.architectural_weights.items():
            idx = config[arch_key]
            arch_weight.data[idx] = 1
        arch_info = self.query()
        y = -arch_info['cifar10-valid']['valid_accuracy']
        c = arch_info['cifar10-valid']['latency (ms)']
        return {
            'config': config,
            'loss': y,
            'cost': c,
            'status': STATUS_OK}

    #NOTE: this works only for nasbench201 for now
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



