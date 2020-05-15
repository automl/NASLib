import numpy as np
import torch

from naslib.optimizers.discrete.rs import RandomSearch
from naslib.optimizers.core.operations import CategoricalOp

class RegularizedEvolution(RandomSearch):
    def __init__(self, *args, **kwargs):
        super(RegularizedEvolution, self).__init__(*args, **kwargs)

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

    def mutate_arch(self, parent_arch):
        self.set_to_zero()

        dim = np.random.choice(list(parent_arch))
        arch_weight = parent_arch[dim]

        argmax = int(arch_weight.argmax().data.numpy())
        list_of_idx = list(range(len(arch_weight)))
        list_of_idx.remove(argmax)
        idx = np.random.choice(list_of_idx)
        parent_arch[dim].data[argmax] = 0
        parent_arch[dim].data[idx].data = 1

        self.architectural_weights = parent_arch
