import logging

import torch

from naslib.optimizers.core.operations import GDASMixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer


class GDASOptimizer(DARTSOptimizer):
    """
    Implements GDAS as defined in

        Dong and Yang (2019): Searching for a Robust Neural Architecture in Four GPU Hours

    """
    def __init__(self, 
            epochs: int, 
            tau_max: float = 10, 
            tau_min: float = 0.1, 
            op_optimizer: torch.optim.Optimizer = torch.optim.SGD, 
            arch_optimizer: torch.optim.Optimizer = torch.optim.Adam, 
            loss_criteria=torch.nn.CrossEntropyLoss(), 
            grad_clip=None
        ):
        """
        Instantiate the optimizer

        Args:
            epochs (int): Number of epochs. Required for tau
            tau_max (float): Initial tau
            tau_min (float): The minimum tau where it is decayed to
            op_optimizer (torch.optim.Optimizer): optimizer for the op weights
            arch_optimizer (torch.optim.Optimizer): optimizer for the architecture weights
            loss_criteria: The loss.
            grad_clip (float): Clipping of the gradients. Default None.
        """
        super(GDASOptimizer, self).__init__(op_optimizer, arch_optimizer, 
            loss_criteria, grad_clip)

        self.epochs = epochs
        self.tau_max = tau_max
        self.tau_min = tau_min

        # Linear tau schedule
        self.tau_step = (self.tau_min - self.tau_max) / self.epochs
        self.tau_curr = self.tau_max

    
    @staticmethod
    def update_tau(current_edge_data, tau):
        """
        Add tau as a shared attribute to edges.
        """
        if current_edge_data.has('final') and current_edge_data.final:
            return current_edge_data
        current_edge_data.set('tau', tau, shared=True)
        return current_edge_data

    
    @staticmethod
    def update_ops(current_edge_data):
        """
        Function to replace the primitive ops at the edges
        with the GDAS specific GDASMixedOp.
        """
        if current_edge_data.has('final') and current_edge_data.final:
            return current_edge_data
        primitives = current_edge_data.op
        current_edge_data.set('op', GDASMixedOp(primitives))
        return current_edge_data


    def adapt_search_space(self, search_space, scope=None):
        """
        Adapts the search space for GDAS. Use same as in DARTS.

        tau does not need to be set here, as `new_epoch()` will
        be called before the first forward pass.
        """
        super().adapt_search_space(search_space, scope)

        self.scope = scope if scope else self.graph.OPTIMIZER_SCOPE


    def new_epoch(self, epoch):
        """
        Update the tau softmax parameter at the edges.

        This is also initially called before epoch 1.
        """
        super(GDASOptimizer, self).new_epoch(epoch)
        
        self.tau_curr += self.tau_step
        self.graph.update_edges(
            lambda current_edge_data: self.update_tau(current_edge_data, tau=self.tau_curr),
            scope=self.scope,
            private_edge_data=False
        )

        logging.info('TAU {}'.format(self.tau_curr))






    # def replace_function(self, edge, graph):
    #     graph.architectural_weights = self.architectural_weights

    #     if 'op_choices' in edge:
    #         edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

    #         weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
    #             torch.nn.Parameter(1e-3 * torch.randn(size=[len(edge['op_choices'])], requires_grad=True))

    #         self.architectural_weights[edge_key] = weights
    #         edge['arch_weight'] = self.architectural_weights[edge_key]
    #         edge['op'] = GDASMixedOp(primitives=edge['op_choices'], **edge['op_kwargs'])

    #         if edge_key not in self.edges:
    #             self.edges[edge_key] = []
    #         self.edges[edge_key].append(edge)
    #     return edge

    # def forward_pass_adjustment(self, *args, **kwargs):
    #     """
    #     Replaces the architectural weights in the edges with gumbel softmax near one-hot encodings.
    #     """

    #     for arch_key, arch_weight in self.architectural_weights.items():
    #         # gumbel sample arch weights and assign them in self.edges
    #         sampled_arch_weight = torch.nn.functional.gumbel_softmax(
    #             arch_weight, tau=self.tau_curr, hard=False
    #         )

    #         # random perturbation part
    #         if self.perturb_alphas == 'random':
    #             softmaxed_arch_weight = sampled_arch_weight.clone()
    #             perturbation = torch.zeros_like(softmaxed_arch_weight).uniform_(
    #                 -self.epsilon_alpha,
    #                 self.epsilon_alpha
    #             )
    #             softmaxed_arch_weight.data.add_(perturbation)
    #             # clipping
    #             max_index = softmaxed_arch_weight.argmax()
    #             softmaxed_arch_weight.data.clamp_(0, 1)
    #             if softmaxed_arch_weight.sum() == 0.0:
    #                 softmaxed_arch_weight.data[max_index] = 1.0
    #             softmaxed_arch_weight.data.div_(softmaxed_arch_weight.sum())

    #         for edge in self.edges[arch_key]:
    #             edge['sampled_arch_weight'] = sampled_arch_weight
    #             if self.perturb_alphas == 'random':
    #                 edge['softmaxed_arch_weight'] = softmaxed_arch_weight
    #                 edge['perturb_alphas'] = True

    # @classmethod
    # def from_config(cls, *args, **kwargs):
    #     nas_opt = cls(*args, **kwargs)
    #     return nas_opt