from abc import abstractmethod

import torch

from naslib.search_spaces.core.operations import CategoricalOp, MixedOp


class MetaOptimizer(torch.nn.Module):
    def __init__(self):
        super(MetaOptimizer, self).__init__()

    @abstractmethod
    def replace_function(self, edge, graph):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        pass


class OneShotOptimizer(MetaOptimizer):
    def __init__(self):
        super(OneShotOptimizer).__init__()

    def replace_function(self, edge, graph):
        if 'op_choices' in edge:
            edge['op'] = CategoricalOp(primitives=edge['op_choices'], **edge['op_kwargs'])
        return edge


class DARTSOptimizer(MetaOptimizer):
    def __init__(self):
        super(DARTSOptimizer, self).__init__()
        self.architectural_weights = torch.nn.ParameterDict()

    def replace_function(self, edge, graph):
        if 'op_choices' in edge:
            edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

            weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
                torch.nn.Parameter(torch.randn(size=[len(edge['op_choices'])], requires_grad=True))

            self.architectural_weights[edge_key] = weights
            edge['op'] = MixedOp(primitives=edge['op_choices'], weights=weights, **edge['op_kwargs'])
        return edge
