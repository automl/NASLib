import numpy as np
import torch
import logging
from torch.autograd import Variable
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer
from naslib.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

import naslib.search_spaces.core.primitives as ops

logger = logging.getLogger(__name__)


class DrNASGrowOptimizer(DARTSOptimizer):
    """
    Implementation of DrNAS optimizer introduced in the paper
        DrNAS: Dirichlet Neural Architecture Search (ICLR2021)
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
    def sample_alphas(edge):
        # ? check if we need to unsqueeze here? -- torch.unsqueeze(edge.data.alpha, dim=0)
        beta = F.elu(edge.data.alpha) + 1
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()
        edge.data.set("sampled_arch_weight", weights, shared=True)

    @staticmethod
    def remove_sampled_alphas(edge):
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the DrNAS specific DrNASMixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", DrNASMixedOp(primitives))

    def __init__(
        self,
        config,
        op_optimizer=torch.optim.SGD,
        arch_optimizer=torch.optim.Adam,
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Initialize a new instance.

        Args:

        """
        super().__init__(config, op_optimizer, arch_optimizer, loss_criteria)
        self.reg_type = "kl"
        self.reg_scale = 1e-3
        # self.reg_scale = config.reg_scale
        self.epochs = config.search.epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def new_epoch(self, epoch):
        super().new_epoch(epoch)

    def adapt_search_space(self, search_space, scope=None):
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
        logger.info("Parsed graph:\n" + graph.modules_str())

        # Init optimizers
        if self.arch_optimizer is not None:
            self.arch_optimizer = self.arch_optimizer(
                self.architectural_weights.parameters(),
                lr=self.config.search.arch_learning_rate,
                betas=(0.5, 0.999),
                weight_decay=self.config.search.arch_weight_decay,
            )

        self.op_optimizer = self.op_optimizer(
            graph.parameters(),
            lr=self.config.search.learning_rate,
            momentum=self.config.search.momentum,
            weight_decay=self.config.search.weight_decay,
        )

        graph.train()

        self.graph = graph
        self.scope = scope

        self.anchor = Dirichlet(
            torch.ones_like(
                torch.nn.utils.parameters_to_vector(self.architectural_weights)
            ).to(self.device)
        )

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        # sample weights (alphas) from the dirichlet distribution (parameterized by beta) and set to edges
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge),
            scope=self.scope,
            private_edge_data=False,
        )

        # Update architecture weights
        self.arch_optimizer.zero_grad()
        logits_val = self.graph(input_val)
        val_loss = self.loss(logits_val, target_val)

        if self.reg_type == "kl":
            val_loss += self._get_kl_reg()

        val_loss.backward()

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.architectural_weights.parameters(), self.grad_clip
            )
        self.arch_optimizer.step()

        # has to be done again, cause val_loss.backward() frees the gradient from sampled alphas
        # TODO: this is not how it is intended because the samples are now different. Another
        # option would be to set val_loss.backward(retain_graph=True) but that requires more memory.
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge),
            scope=self.scope,
            private_edge_data=False,
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
            private_edge_data=False,
        )

        return logits_train, logits_val, train_loss, val_loss

    def _get_kl_reg(self):
        cons = (
            F.elu(torch.nn.utils.parameters_to_vector(self.architectural_weights)) + 1
        )
        q = Dirichlet(cons)
        p = self.anchor
        kl_reg = self.reg_scale * torch.sum(kl_divergence(q, p))
        return kl_reg

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


class DrNASMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed ops

        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(primitives)
        self.min_cuda_memory = min_cuda_memory

    def get_weights(self, edge_data):
        return edge_data.sampled_arch_weight

    def process_weights(self, weights):
        return weights

    def apply_weights(self, x, weights):
        weighted_sum = sum(
            w * op(x, None)
            for w, op in zip(weights, self.primitives)
        )
        return weighted_sum