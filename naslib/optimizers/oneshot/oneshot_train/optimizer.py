import torch
import logging
import numpy as np

from naslib.search_spaces.core.primitives import MixedOp
from naslib.search_spaces.nasbench301.conversions import Genotype
from naslib.optimizers import DARTSOptimizer

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
        with torch.no_grad():
            alpha = torch.nn.Parameter(
                torch.ones(size=[len_primitives], requires_grad=False)
            )
        edge.data.set("alpha", alpha, shared=True)

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the OneShotOp, which is just a summation of these ops.
        """
        primitives = edge.data.op
        edge.data.set("op", OneShotOp(primitives))

    def __init__(
        self,
        learning_rate: float = 0.025,
        momentum: float = 0.9,
        weight_decay: float = 0.0003,
        grad_clip: int = 5,
        unrolled: bool = False,
        arch_learning_rate: float = 0.0003,
        arch_weight_decay: float = 0.001,
        epochs: int = 50,
        op_optimizer: str = 'SGD',
        arch_optimizer: str = 'Adam',
        loss_criteria: str = 'CrossEntropyLoss',
        **kwargs
    ):

        super().__init__(learning_rate, momentum, weight_decay, grad_clip, unrolled, arch_learning_rate, arch_weight_decay, op_optimizer, arch_optimizer, loss_criteria)

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        logits_val = self.graph(input_val)
        val_loss = self.loss(logits_val, target_val)
        val_loss.backward()

        # Update op weights
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        return logits_train, logits_val, train_loss, val_loss

    def set_alphas_from_path(self, arch_encoding):
        """
        arch_encoding: this can be either a Genotype object (when the nasbench301
        space) or a list of 6 integers (when the nb201 space), aka op_indices
        """

        if self.graph.get_type() == "nasbench201":
            assert type(arch_encoding) in [
                list,
                np.ndarray,
            ], "nasbench201 requires a list of ints of size 6 in order to query the one-shot model."

            with torch.no_grad():
                for i, op_index in enumerate(arch_encoding):
                    _new_alpha = torch.nn.Parameter(
                        torch.zeros(size=[5], requires_grad=False)
                    )
                    _new_alpha[op_index] = 1
                    self.architectural_weights[i].copy_(_new_alpha)

        elif self.graph.get_type() == "nasbench301":
            assert (
                type(arch_encoding) is Genotype
            ), "darts requires a Genotype object in order to query the one-shot model."

            def update_alphas(cell_type, alphas):
                n_inputs = 2
                start_idx = 0
                end_idx = 2

                for i, (op, input_node) in enumerate(cell_type):
                    if i % 2 == 0:
                        alphas_subset = alphas[start_idx:end_idx]
                        n_inputs += 1
                        start_idx = end_idx
                        end_idx += n_inputs

                    alphas_subset[input_node][ops.index(op)] = 1

            # darts = [id, zero, maxpool, avg, sep3, sep5, dil3, dil5]
            ops = [
                "skip_connect",
                "zero",
                "max_pool_3x3",
                "avg_pool_3x3",
                "sep_conv_3x3",
                "sep_conv_5x5",
                "dil_conv_3x3",
                "dil_conv_5x5",
            ]

            # set all alphas to 0 firstly
            with torch.no_grad():
                for alpha in self.architectural_weights:
                    alpha.copy_(
                        torch.nn.Parameter(
                            torch.zeros(size=[len(ops)], requires_grad=False)
                        )
                    )

                update_alphas(
                    arch_encoding.normal, self.graph.get_all_edge_data("alpha")[:14]
                )
                update_alphas(
                    arch_encoding.reduce, self.graph.get_all_edge_data("alpha")[14:]
                )

    def get_final_architecture(self):
        # TODO
        # for using the one-shot model as performance predictor it is not
        # necessary
        return NotImplementedError


class OneShotOp(MixedOp):
    """
    One-Shot representation of the discrete search space.
    """

    def __init__(self, primitives):
        super().__init__(primitives)

    def get_weights(self, edge_data):
        return edge_data.alpha

    def process_weights(self, weights):
        return weights

    def apply_weights(self, x, weights):
        return sum(w * op(x, None) for w, op in zip(weights, self.primitives))
