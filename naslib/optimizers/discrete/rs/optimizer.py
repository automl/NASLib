import numpy as np
import torch

from naslib.optimizers.core import NASOptimizer
from naslib.optimizers.core.operations import CategoricalOp

class RandomSearch(NASOptimizer):
    def __init__(self, *args, **kwargs):
        super(RandomSearch, self).__init__()
        self.architectural_weights = torch.nn.ParameterDict()

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def replace_function(self, edge, graph):
        graph.architectural_weights = self.architectural_weights

        if 'op_choices' in edge:
            edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

            weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
                torch.nn.Parameter(torch.zeros(size=[len(edge['op_choices'])],
                                               requires_grad=False))

            self.architectural_weights[edge_key] = weights
            edge['arch_weight'] = self.architectural_weights[edge_key]
            edge['op'] = CategoricalOp(primitives=edge['op_choices'], **edge['op_kwargs'])

        return edge

    def uniform_sample(self, *args, **kwargs):
        self.set_to_zero()
        for arch_key, arch_weight in self.architectural_weights.items():
            idx = np.random.choice(len(arch_weight))
            arch_weight.data[idx] = 1

    def set_to_zero(self, *args, **kwargs):
        for arch_key, arch_weight in self.architectural_weights.items():
            arch_weight.data = torch.zeros(size=[len(arch_weight)])

    def step(self, *args, **kwargs):
        self.uniform_sample()

