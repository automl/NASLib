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
        super().__init__(config, op_optimizer, arch_optimizer, loss_criteria)

        self.epochs = config.epochs
        self.tau_max = config.tau_max
        self.tau_min = config.tau_min

        # Linear tau schedule
        self.tau_step = (self.tau_min - self.tau_max) / self.epochs
        self.tau_curr = self.tau_max
    
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


    def new_epoch(self, epoch):
        """
        Update the tau softmax parameter at the edges.

        This is also initially called before epoch 1.
        """
        super().new_epoch(epoch)
        
        self.tau_curr += self.tau_step
        logger.info("tau {}".format(self.tau_curr))
    

    @staticmethod
    def sample_alphas(current_edge_data, tau):
        if current_edge_data.has('final') and current_edge_data.final:
            return current_edge_data
        sampled_arch_weight = torch.nn.functional.gumbel_softmax(
            current_edge_data.alpha, tau=tau, hard=True
        )
        current_edge_data.set('sampled_arch_weight', sampled_arch_weight, shared=True)
        return current_edge_data
    

    @staticmethod
    def remove_sampled_alphas(current_edge_data):
        if current_edge_data.has('sampled_arch_weight'):
            current_edge_data.remove('sampled_arch_weight')
        return current_edge_data

    
    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        # sample alphas and set to edges
        self.graph.update_edges(
            update_func=lambda current_edge_data: self.sample_alphas(current_edge_data, self.tau_curr),
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
            update_func=lambda current_edge_data: self.sample_alphas(current_edge_data, self.tau_curr),
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