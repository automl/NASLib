import logging

import torch

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer
import ProxSGD_for_groups as ProxSGD
import utils_sparsenas

logger = logging.getLogger(__name__)


class GSparseOptimizer(DARTSOptimizer):
    """
    Implements Group Sparsity as defined in

        GSparsity: Unifying Network Pruning and 
        Neural Architecture Search by Group Sparsity
    """
    def __init__(
        self,
        config,
        op_optimizer: torch.optim.Optimizer = torch.optim.SGD,
        arch_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Instantiate the optimizer

        Args:
            epochs (int): Number of epochs. Required for tau
            mu (float): corresponds to the Weight decay
            threshold (float): threshold of pruning
            op_optimizer (torch.optim.Optimizer): optimizer for the op weights
            arch_optimizer (torch.optim.Optimizer): optimizer for the architecture weights
            loss_criteria: The loss.
            grad_clip (float): Clipping of the gradients. Default None.
        """
        super().__init__(config, op_optimizer, arch_optimizer, loss_criteria)

        self.grad_clip = config.search.grad_clip
        self.threshold = config.search.threshold
        self.mu = config.search.mu

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the GSparse specific GSparseMixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", GSparseMixedOp(primitives))



class GSparseMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed op for Group Sparsity.

        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(primitives)
        self.min_cuda_memory = min_cuda_memory

    def forward(self, x, edge_data):
        """
        Applies the gumbel softmax to the architecture weights
        before forwarding `x` through the graph as in DARTS
        """
        # sampled_arch_weight = edge_data.sampled_arch_weight
        # result1 = sum(w * op(x, None) for w, op in zip(sampled_arch_weight, self.primitives))

        summed = sum(op(x, None) for op in self.primitives)

        return summed