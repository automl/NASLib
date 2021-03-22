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


class PCDARTSOptimizer(DARTSOptimizer):
    """
    Implementation of PC-DARTS optimizer introduced in the paper
    PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search (ICLR2020)
    note: many functions are similar to the DARTS optimizer so it makes sense to inherit this class directly 
    from DARTSOptimizer instead of MetaOptimizer
    """
    """
    questions:

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
    def sample_alphas(edge):
        #? check if we need to unsqueeze here? -- torch.unsqueeze(edge.data.alpha, dim=0)
        beta = F.elu(edge.data.alpha) + 1
        # what to do about masking and pruning?
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()
        edge.data.set('sampled_arch_weight', weights, shared = True)
        # check if argmax is being used somewhere else

    @staticmethod
    def remove_sampled_alphas(edge):
      if (edge.data.has('sampled_arch_weight')):
        edge.data.remove('sampled_arch_weight')
    
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
        
        ## check if 'beta' must be added as a parameter here
        self.epochs = config.search.epochs

    def new_epoch(self, epoch):
      super().new_epoch(epoch)


    def adapt_search_space(self, search_space, scope=None):
        """
        Same as in darts with a different mixop.
        Just add dirichlet parameter as buffer so it is checkpointed.
        """
        super().adapt_search_space(search_space, scope)

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        # sample alphas and set to edges
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge),
            scope=self.scope,
            private_edge_data=False
        )
        
        # Update architecture weights
        self.arch_optimizer.zero_grad()
        logits_val = self.graph(input_val)
        val_loss = self.loss(logits_val, target_val)
        val_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.architectural_weights.parameters(), self.grad_clip)
        self.arch_optimizer.step()

        # has to be done again, cause val_loss.backward() frees the gradient from sampled alphas
        # TODO: this is not how it is intended because the samples are now different. Another 
        # option would be to set val_loss.backward(retain_graph=True) but that requires more memory.
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge),
            scope=self.scope,
            private_edge_data=False
        )

        # Update op weights
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        # in order to properly unparse remove the alphas again
        self.graph.update_edges(
            update_func=self.remove_sampled_alphas,
            scope=self.scope,
            private_edge_data=False
        )
        
        return logits_train, logits_val, train_loss, val_loss

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
        applies the previously sampled weights from the dirichlet distribution
        before forwarding `x` through the graph as in DARTS
        """
        weigsum = sum(w * op(x, None) for w, op in zip(edge_data.sampled_arch_weight, self.primitives))
        return weigsum
    
    def get_embedded_ops(self):
        return self.primitives