import logging

import torch

from naslib.optimizers.core.operations import GDASMixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer

logger = logging.getLogger(__name__)

class GDASOptimizer(DARTSOptimizer):
    """
    Implements GDAS as defined in

        Dong and Yang (2019): Searching for a Robust Neural Architecture in Four GPU Hours

    """
    def __init__(self, config,
            op_optimizer: torch.optim.Optimizer = torch.optim.SGD, 
            arch_optimizer: torch.optim.Optimizer = torch.optim.Adam, 
            loss_criteria=torch.nn.CrossEntropyLoss()
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
        super(GDASOptimizer, self).__init__(config, op_optimizer, arch_optimizer, 
            loss_criteria)

        self.epochs = config.epochs
        self.tau_max = config.tau_max
        self.tau_min = config.tau_min

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

        logging.info("tau {}".format(self.tau_curr))