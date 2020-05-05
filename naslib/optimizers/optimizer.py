from abc import ABCMeta, abstractmethod

import six
import torch

from naslib.search_spaces.core.operations import CategoricalOp, MixedOp


@six.add_metaclass(ABCMeta)
class MetaOptimizer:
    def __init__(self):
        super(MetaOptimizer, self).__init__()

    @abstractmethod
    def replace_function(self, edge, graph):
        raise NotImplementedError


class OneShotOptimizer(MetaOptimizer):
    def __init__(self):
        super(OneShotOptimizer).__init__()

    def replace_function(self, edge, graph):
        if 'op_choices' in edge:
            edge['op'] = CategoricalOp(primitives=edge['op_choices'], **edge['op_kwargs'])
        return edge


class DARTSOptimizer(MetaOptimizer):
    def __init__(self):
        super(DARTSOptimizer).__init__()
        self.architectural_weights = {
            'normal': {},
            'reduction': {}
        }

    def replace_function(self, edge, graph):
        if 'op_choices' in edge:
            cell_type_weights = self.architectural_weights[graph.cell_type]

            edge_key = 'from_{}_to_{}'.format(edge['from_node'], edge['to_node'])

            weights = cell_type_weights.get(edge_key, torch.randn(size=[len(edge['op_choices'])], requires_grad=True))
            self.architectural_weights[graph.cell_type][edge_key] = weights
            edge['op'] = MixedOp(primitives=edge['op_choices'], weights=weights, **edge['op_kwargs'])
        return edge
