import numpy as np
import torch
import logging
from torch.autograd import Variable

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils.pytorch_helper import create_optimizer, create_criterion

import naslib.search_spaces.core.primitives as ops

logger = logging.getLogger(__name__)


class DARTSOptimizer(MetaOptimizer):
    """
    Implementation of the DARTS paper as in
        Liu et al. 2019: DARTS: Differentiable Architecture Search.
    """

    @staticmethod
    def add_alphas(edge):
        """
        Function to add the architectural weights to the edges.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(
            1e-3 * torch.randn(size=[len_primitives], requires_grad=True)
        )
        edge.data.set("alpha", alpha, shared=True)

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the DARTS specific MixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", DARTSMixedOp(primitives))

    def __init__(
        self,
        learning_rate: float = 0.025,
        momentum: float = 0.9,
        weight_decay: float = 0.0003,
        grad_clip: int = 5,
        unrolled: bool = False,
        arch_learning_rate: float = 0.0003,
        arch_weight_decay: float = 0.001,
        op_optimizer: str = 'SGD',
        arch_optimizer: str = 'Adam',
        loss_criteria: str = 'CrossEntropyLoss',
        **kwargs,
    ):
        """
        Initialize a new instance.

        Args:

        """
        super(DARTSOptimizer, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.unrolled = unrolled
        self.arch_learning_rate = arch_learning_rate
        self.arch_weight_decay = arch_weight_decay

        self.op_optimizer = op_optimizer
        self.arch_optimizer = arch_optimizer
        self.loss = loss_criteria

        self.architectural_weights = torch.nn.ParameterList()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.perturb_alphas = None
        self.epsilon = 0

    def adapt_search_space(self, search_space, dataset, scope=None, **kwargs):
        # We are going to modify the search space
        self.search_space = search_space
        graph = search_space.clone()

        # If there is no scope defined, let's use the search space default one
        if not scope:
            scope = graph.OPTIMIZER_SCOPE

        # 1. add alphas
        graph.update_edges(
            self.__class__.add_alphas, scope=scope, private_edge_data=False
        )

        # 2. replace primitives with mixed_op
        graph.update_edges(
            self.__class__.update_ops, scope=scope, private_edge_data=True
        )

        for alpha in graph.get_all_edge_data("alpha"):
            self.architectural_weights.append(alpha)

        graph.parse()
        #logger.info("Parsed graph:\n" + graph.modules_str())

        # Init optimizers
        self.arch_optimizer = create_optimizer(
            opt=self.arch_optimizer,
            params=self.architectural_weights.parameters(),
            lr=self.arch_learning_rate,
            weight_decay=self.arch_weight_decay,
            betas=(0.5, 0.999)
        )

        self.op_optimizer = create_optimizer(
            opt=self.op_optimizer,
            params=graph.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.loss = create_criterion(
            crit=self.loss,
        )

        graph.train()

        self.graph = graph
        self.scope = scope
        self.dataset = dataset

    def get_checkpointables(self):
        return {
            "model": self.graph,
            "op_optimizer": self.op_optimizer,
            "arch_optimizer": self.arch_optimizer,
            "arch_weights": self.architectural_weights,
        }

    def before_training(self):
        """
        Move the graph into cuda memory if available.
        """
        self.graph = self.graph.to(self.device)
        self.architectural_weights = self.architectural_weights.to(self.device)

    def new_epoch(self, epoch):
        """
        Just log the architecture weights.
        """
        alpha_str = [
            ", ".join(["{:+.06f}".format(x) for x in a])
            + ", {}".format(np.argmax(a.detach().cpu().numpy()))
            for a in self.architectural_weights
        ]
        logger.info(
            "Arch weights (alphas, last column argmax): \n{}".format(
                "\n".join(alpha_str)
            )
        )
        super().new_epoch(epoch)

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        unrolled = False  # what it this?

        if unrolled:
            raise NotImplementedError()
        else:
            # Update architecture weights
            self.arch_optimizer.zero_grad()
            logits_val = self.graph(input_val)
            val_loss = self.loss(logits_val, target_val)
            val_loss.backward()

            self.arch_optimizer.step()

            # Update op weights
            self.op_optimizer.zero_grad()
            logits_train = self.graph(input_train)
            train_loss = self.loss(logits_train, target_train)
            train_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
            self.op_optimizer.step()

        return logits_train, logits_val, train_loss, val_loss

    def get_final_architecture(self):
        logger.info(
            "Arch weights before discretization: {}".format(
                [a for a in self.architectural_weights]
            )
        )
        graph = self.graph.clone().unparse()
        graph.prepare_discretization()

        def discretize_ops(edge):
            if edge.data.has("alpha"):
                primitives = edge.data.op.get_embedded_ops()
                alphas = edge.data.alpha.detach().cpu()
                edge.data.set("op", primitives[np.argmax(alphas)])

        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)
        graph.prepare_evaluation()
        graph.parse()
        graph = graph.to(self.device)
        return graph

    def get_op_optimizer(self):
        return self.op_optimizer.__class__

    def get_model_size(self):
        return count_parameters_in_MB(self.graph)

    def test_statistics(self):
        # nb301 is not there but we use it anyways to generate the arch strings.
        # if self.graph.QUERYABLE:
        try:
            # record anytime performance
            best_arch = self.get_final_architecture()
            return best_arch.query(Metric.TEST_ACCURACY, self.dataset)
        except:
            return None

    def _step(
        self,
        model,
        criterion,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
        unrolled,
    ):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                model,
                criterion,
                input_train,
                target_train,
                input_valid,
                target_valid,
                eta,
                network_optimizer,
            )
        else:
            self._backward_step(model, criterion, input_valid, target_valid)

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.architectural_weights.parameters(), self.grad_clip
            )
        self.optimizer.step()

    def _backward_step(self, model, criterion, input_valid, target_valid):
        """Compute 1st order approximation"""
        loss = self._loss(model, criterion, input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(
        self,
        model,
        criterion,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
    ):
        raise NotImplementedError

    def _loss(self, model, criterion, input, target):
        pred = model(input)
        return criterion(pred, target)


class DARTSMixedOp(MixedOp):
    """
    Continous relaxation of the discrete search space.
    """

    def __init__(self, primitives):
        super().__init__(primitives)
    
    def get_weights(self, edge_data):
        return edge_data.alpha
    
    def process_weights(self, weights):
        return torch.softmax(weights, dim=-1)

    def apply_weights(self, x, weights):        
        return sum(w * op(x, None) for w, op in zip(weights, self.primitives))

