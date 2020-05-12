import torch

from naslib.optimizers.core.operations import PCDARTSMixedOp
from naslib.optimizers.oneshot.darts import DARTSOptimizer


class PCDARTSOptimizer(DARTSOptimizer):
    def __init__(self, channel_divisor, *args, **kwargs):
        super(PCDARTSOptimizer, self, ).__init__(*args, **kwargs)
        self.channel_divisor = channel_divisor

    def replace_function(self, edge, graph):
        graph.architectural_weights = self.architectural_weights

        if 'op_choices' in edge:
            edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

            weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
                torch.nn.Parameter(1e-3 * torch.randn(size=[len(edge['op_choices'])], requires_grad=True))

            self.architectural_weights[edge_key] = weights
            edge['arch_weight'] = self.architectural_weights[edge_key]
            edge['op'] = PCDARTSMixedOp(primitives=edge['op_choices'], channel_divisor=self.channel_divisor,
                                        **edge['op_kwargs'])

            if edge_key not in self.edges:
                self.edges[edge_key] = []
            self.edges[edge_key].append(edge)
        return edge

