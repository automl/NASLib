import numpy as np
import torch
import logging
from torch.autograd import Variable

from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.darts.conversions import Genotype
from naslib.optimizers import DARTSOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

import naslib.search_spaces.core.primitives as ops

logger = logging.getLogger(__name__)


class OneShotNASOptimizer(DARTSOptimizer):
    """
    Implementation of the One-Shot NAS training as in
        Bender et al. 2018: Understanding and Simplifying One-Shot Neural Architecture Search.
    """

    @staticmethod
    def add_alphas(edge):
        """
        Function to add the architectural weights to the edges.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(torch.ones(size=[len_primitives],
                                              requires_grad=False))
        edge.data.set('alpha', alpha, shared=True)


    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the OneShotOp, which is just a summation of these ops.
        """
        primitives = edge.data.op
        edge.data.set('op', OneShotOp(primitives))


    def __init__(self, config,
            op_optimizer=torch.optim.SGD,
            arch_optimizer=None,
            loss_criteria=torch.nn.CrossEntropyLoss()):

        super(OneShotNASOptimizer, self).__init__(config, op_optimizer, arch_optimizer,
                                                  loss_criteria)


    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        logits_val = self.graph(input_val)
        val_loss = self.loss(logits_val, target_val)

        # Update op weights
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        return logits_train, logits_val, train_loss, val_loss


    def query_single_path(self, arch_encoding):
        """
        arch_encoding: this can be either a Genotype object (when the darts
        space) or a list of 6 integers (when the nb201 space), aka op_indices
        """

        if self.graph.get_type() == 'nasbench201':
            assert type(arch_encoding) is list, "nasbench201 requires a list of ints of size 6 in order to query the one-shot model."

            for i, op_index in enumerate(arch_encoding):
                _new_alpha = torch.nn.Parameter(torch.zeros(size=[5], requires_grad=False))
                _new_alpha[op_index] = 1
                self.architectural_weights[i].detach_().copy_(_new_alpha)

        elif self.graph.get_type() == 'darts':
            assert type(arch_encoding) is Genotype, "darts requires a Genotype object in order to query the one-shot model."


    def get_final_architecture(self):
        #TODO
        # for using the one-shot model as performance predictor it is not
        # necessary
        return NotImplementedError


class OneShotOp(AbstractPrimitive):
    """
    One-Shot representation of the discrete search space.
    """
    def __init__(self, primitives):
        super().__init__(locals())
        self.primitives = primitives
        for i, primitive in enumerate(primitives):
            self.add_module("primitive-{}".format(i), primitive)

    def forward(self, x, edge_data):
        """
        Element-wise summation of the output tensors coming from each edge.
        """
        return sum(w * op(x, None) for w, op in zip(edge_data.alpha, self.primitives))

    def get_embedded_ops(self):
        return self.primitives
