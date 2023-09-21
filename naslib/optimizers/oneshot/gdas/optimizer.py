import logging

import torch

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer

logger = logging.getLogger(__name__)


class GDASOptimizer(DARTSOptimizer):
    """
    Implements GDAS as defined in

        Dong and Yang (2019): Searching for a Robust Neural Architecture in Four GPU Hours

    """

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
        epochs: int = 50,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        **kwargs,
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
        super().__init__(learning_rate, momentum, weight_decay, grad_clip, unrolled, arch_learning_rate, arch_weight_decay, op_optimizer, arch_optimizer, loss_criteria)

        self.epochs = epochs
        self.tau_max = tau_max
        self.tau_min = tau_min

        # Linear tau schedule
        self.tau_step = (self.tau_min - self.tau_max) / self.epochs
        self.tau_curr = torch.Tensor([self.tau_max])  # make it checkpointable

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the GDAS specific GDASMixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", GDASMixedOp(primitives))

    def adapt_search_space(self, search_space, dataset, scope=None):
        """
        Same as in darts with a different mixop.
        Just add tau as buffer so it is checkpointed.
        """
        super().adapt_search_space(search_space, dataset, scope)
        self.graph.register_buffer("tau", self.tau_curr)

    def new_epoch(self, epoch):
        """
        Update the tau softmax parameter at the edges.

        This is also initially called before epoch 1.
        """
        super().new_epoch(epoch)

        self.tau_curr += self.tau_step
        logger.info("tau {}".format(self.tau_curr))

    @staticmethod
    def sample_alphas(edge, tau):
        # sampled_arch_weight = torch.nn.functional.gumbel_softmax(
        #     edge.data.alpha, tau=float(tau), hard=True
        # )
        # edge.data.set('sampled_arch_weight', sampled_arch_weight, shared=True)

        # from gdas repo
        # https://github.com/D-X-Y/AutoDL-Projects/blob/befa6bcb00e0a8fcfba447d2a1348202759f58c9/lib/models/cell_searchs/search_model_gdas.py#L88
        # https://github.com/D-X-Y/AutoDL-Projects/blob/befa6bcb00e0a8fcfba447d2a1348202759f58c9/lib/models/cell_searchs/search_cells.py#L51
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch_parameters = torch.unsqueeze(edge.data.alpha, dim=0)

        while True:
            gumbels = -torch.empty_like(arch_parameters).exponential_().log()
            gumbels = gumbels.to(device)
            tau = tau.to(device)
            arch_parameters = arch_parameters.to(device)
            logits = (arch_parameters.log_softmax(dim=1) + gumbels) / tau
            probs = torch.nn.functional.softmax(logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (
                (torch.isinf(gumbels).any())
                or (torch.isinf(probs).any())
                or (torch.isnan(probs).any())
            ):
                continue
            else:
                break

        weights = hardwts[0]
        argmaxs = index[0].item()

        edge.data.set("sampled_arch_weight", weights, shared=True)
        edge.data.set("argmax", argmaxs, shared=True)

    @staticmethod
    def remove_sampled_alphas(edge):
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        # sample alphas and set to edges
        self.graph.update_edges(
            update_func=lambda edge: self.sample_alphas(edge, self.tau_curr),
            scope=self.scope,
            private_edge_data=False,
        )

        # Update architecture weights
        self.arch_optimizer.zero_grad()
        logits_val = self.graph(input_val)
        val_loss = self.loss(logits_val, target_val)
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
            update_func=lambda edge: self.sample_alphas(edge, self.tau_curr),
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


class GDASMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed op for GDAS.

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
        """
        Applies the gumbel softmax to the architecture weights
        before forwarding `x` through the graph as in DARTS
        """

        argmax = torch.argmax(weights)

        weighted_sum = sum(
            weights[i] * op(x, None) if i == argmax else weights[i]
            for i, op in enumerate(self.primitives)
        )

        return weighted_sum
