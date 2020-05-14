import logging

import torch

from naslib.optimizers.core.operations import GDASMixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer


class GDASOptimizer(DARTSOptimizer):
    def __init__(self, tau_max, tau_min, *args, **kwargs):
        super(GDASOptimizer, self).__init__(*args, **kwargs)

        # Linear tau schedule
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.tau_step = (self.tau_min - self.tau_max) / self.epochs
        self.tau_curr = self.tau_max

    def new_epoch(self, epoch):
        super(GDASOptimizer, self).new_epoch(epoch)
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
            sampled_arch_weight = torch.nn.functional.gumbel_softmax(
                arch_weight, tau=self.tau_curr, hard=False
            )

            # random perturbation part
            if self.perturb_alphas == 'random':
                softmaxed_arch_weight = sampled_arch_weight.clone()
                perturbation = torch.zeros_like(softmaxed_arch_weight).uniform_(
                    -self.epsilon_alpha,
                    self.epsilon_alpha
                )
                softmaxed_arch_weight.data.add_(perturbation)
                # clipping
                max_index = softmaxed_arch_weight.argmax()
                softmaxed_arch_weight.data.clamp_(0, 1)
                if softmaxed_arch_weight.sum() == 0.0:
                    softmaxed_arch_weight.data[max_index] = 1.0
                softmaxed_arch_weight.data.div_(softmaxed_arch_weight.sum())

            for edge in self.edges[arch_key]:
                edge['sampled_arch_weight'] = sampled_arch_weight
                if self.perturb_alphas == 'random':
                    edge['softmaxed_arch_weight'] = softmaxed_arch_weight
                    edge['perturb_alphas'] = True
