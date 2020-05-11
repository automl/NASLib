import logging

import torch

from naslib.optimizers.core.operations import GDASMixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer


class GDASOptimizer(DARTSOptimizer):
    def __init__(self, tau_max, tau_min, epochs, *args, **kwargs):
        super(GDASOptimizer, self).__init__()
        self.edges = {}

        # Linear tau schedule
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.epochs = epochs
        self.tau_step = (self.tau_min - self.tau_max) / self.epochs
        self.tau_curr = self.tau_max

    def new_epoch(self):
        self.tau_curr += self.tau_step
        logging.info('TAU {}'.format(self.tau_curr))

    def replace_function(self, edge, graph):
        graph.architectural_weights = self.architectural_weights

        if 'op_choices' in edge:
            edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

            weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
                torch.nn.Parameter(1e-3 * torch.randn(size=[len(edge['op_choices'])], requires_grad=True))

            self.architectural_weights[edge_key] = weights
            edge['arch_weight'] = self.architectural_weights[edge_key]
            edge['op'] = GDASMixedOp(primitives=edge['op_choices'], **edge['op_kwargs'])

            if edge_key not in self.edges:
                self.edges[edge_key] = []
            self.edges[edge_key].append(edge)
        return edge

    def forward_pass_adjustment(self, *args, **kwargs):
        """
        Replaces the architectural weights in the edges with gumbel softmax near one-hot encodings.
        """

        for arch_key, arch_weight in self.architectural_weights.items():
            # gumbel sample arch weights and assign them in self.edges
            sampled_arch_weight = torch.nn.functional.gumbel_softmax(arch_weight, tau=self.tau_curr)
            for edge in self.edges[arch_key]:
                edge['sampled_arch_weight'] = sampled_arch_weight
