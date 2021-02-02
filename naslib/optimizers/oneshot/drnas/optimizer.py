import numpy as np
import torch
import logging
from torch.autograd import Variable

from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

import naslib.search_spaces.core.primitives as ops

logger = logging.getLogger(__name__)


class DrNASOptimizer(DARTSOptimizer):
    """
    Implementation of DrNAS optimizer introduced in the paper
        DrNAS: Dirichlet Neural Architecture Search (ICLR2021)
    note: many functions are similar to the DARTS optimizer so it makes sense to inherit this class directly 
    from DARTSOptimizer instead of MetaOptimizer
    """

    @staticmethod
    def add_alphas(edge):
        """
        Function to add the architectural weights to the edges.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(1e-3 * torch.randn(size=[len_primitives], requires_grad=True))
        edge.data.set('alpha', alpha, shared=True)

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the DrNAS specific DrNASMixedOp.
        """
        primitives = edge.data.op
        edge.data.set('op', DrNASMixedOp(primitives))


    def __init__(self, config,
            op_optimizer=torch.optim.SGD, 
            arch_optimizer=torch.optim.Adam, 
            loss_criteria=torch.nn.CrossEntropyLoss()
        ):
        """
        Initialize a new instance.

        Args:
            
        """
        super().__init__(config, op_optimizer, arch_optimizer, loss_criteria)
        
        ## add specifics for the DrNAS optimizer [beta]
        self.epochs = config.search.epochs


    def adapt_search_space(self, search_space, scope=None):
        """
        Same as in darts with a different mixop.
        Just add dirichlet parameter as buffer so it is checkpointed.
        """
        super().adapt_search_space(search_space, scope)

class DrNASMixedOp(AbstractPrimitive):

    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed ops

        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(locals())
        self.primitives = primitives
        for i, primitive in enumerate(primitives):
            self.add_module("primitive-{}".format(i), primitive)


    def forward(self, x, edge_data):
        """
        sample from a dirichlet distribution here and make a fwd pass through the network
        """
        
        return
    
    def get_embedded_ops(self):
        return self.primitives