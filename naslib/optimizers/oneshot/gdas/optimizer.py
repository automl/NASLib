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

    def forward_pass_adjustment(self, *args, **kwargs):
        """
        Replaces the architectural weights in the edges with gumbel softmax near one-hot encodings.
        """

        for arch_key, arch_weight in self.architectural_weights.items():
            # gumbel sample arch weights and assign them in self.edges
            sampled_arch_weight = torch.nn.functional.gumbel_softmax(arch_weight, tau=self.tau_curr)
            for edge in self.edges[arch_key]:
                edge['sampled_arch_weight'] = sampled_arch_weight

