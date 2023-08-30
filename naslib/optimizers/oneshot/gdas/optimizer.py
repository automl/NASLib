import logging

import torch

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.oneshot.darts.optimizer import DARTSOptimizer

logger = logging.getLogger(__name__)


class GDASOptimizer(DARTSOptimizer):
    """
    Implementation of the GDAS optimizer introduced in "Searching for a Robust Neural Architecture in Four GPU Hours"
    by Dong and Yang (2019). Inherits functionalities from DARTSOptimizer and includes additional functionalities
    specific to GDAS.
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
        Initialize a new instance of the GDASOptimizer class.

        Args:
            epochs (int): Total number of training epochs.
            tau_max (float): Initial value of tau.
            tau_min (float): The minimum value to which tau is decayed.
            op_optimizer (str): The optimizer type for operation weights. E.g., 'SGD'
            arch_optimizer (str): The optimizer type for architecture weights. E.g., 'Adam'
            loss_criteria (str): Loss criteria. E.g., 'CrossEntropyLoss'
            grad_clip (float): Clipping of the gradients. Default None.
            **kwargs: Additional keyword arguments.
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
        Replace the primitive operations at the edge with the GDAS-specific GDASMixedOp.

        Args:
            edge: The edge in the computation graph where the operations are to be replaced.
        """
        primitives = edge.data.op
        edge.data.set("op", GDASMixedOp(primitives))

    def adapt_search_space(self, search_space, dataset, scope=None):
        """
        Adapt the search space for GDAS architecture search.

        Args:
            search_space: The initial search space.
            dataset: The dataset for training/validation.
            scope: Scope to update in the search space. Default is None.
        """
        super().adapt_search_space(search_space, dataset, scope)
        self.graph.register_buffer("tau", self.tau_curr)

    def new_epoch(self, epoch):
        """
        Update the tau parameter at the edges at the beginning of each new epoch.

        Args:
            epoch (int): Current epoch number.
        """
        super().new_epoch(epoch)

        self.tau_curr += self.tau_step
        logger.info("tau {}".format(self.tau_curr))

    @staticmethod
    def sample_alphas(edge, tau):
        """
        Sample architecture weights (alphas) using the Gumbel-Softmax distribution parameterized by tau.

        Args:
            edge: The edge in the computation graph where the sampled architecture weights are set.
            tau (torch.Tensor): The tau parameter controlling the temperature of the Gumbel-Softmax distribution.
        """
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
        """
        Remove sampled architecture weights (alphas) from the edge's data.

        Args:
            edge: The edge in the computation graph where the sampled architecture weights are to be removed.
        """
        if edge.data.has("sampled_arch_weight"):
            edge.data.remove("sampled_arch_weight")

    def step(self, data_train, data_val):
        """
        Perform a single optimization step for both architecture and operation weights.

        Args:
            data_train (tuple): Training data as a tuple of inputs and labels.
            data_val (tuple): Validation data as a tuple of inputs and labels.

        Returns:
            tuple: Logits for training data, logits for validation data, loss for training data, loss for validation data.
        """
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
    """
    Specialized MixedOp for GDAS. Handles the sampled architecture weights (alphas).

    """
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize a new instance of the GDASMixedOp class.

        Args:
            primitives (list): List of primitive operations to sample from.
            min_cuda_memory (bool): Whether to minimize CUDA memory usage. Default is False.
        """
        super().__init__(primitives)
        self.min_cuda_memory = min_cuda_memory

    def get_weights(self, edge_data):
        """
        Retrieve the sampled architecture weights from the edge data.

        Args:
            edge_data: The data associated with an edge in the computational graph.

        Returns:
            torch.Tensor: The sampled architecture weights.
        """
        return edge_data.sampled_arch_weight

    def process_weights(self, weights):
        """
        Process the architecture weights if any additional operations are needed.

        Args:
            weights (torch.Tensor): The architecture weights.

        Returns:
            torch.Tensor: The processed architecture weights.
        """
        return weights

    def apply_weights(self, x, weights):
        """
        Apply the architecture weights to the primitive operations.

        Args:
            x (torch.Tensor): Input tensor.
            weights (torch.Tensor): The architecture weights.

        Returns:
            torch.Tensor: The output tensor after applying the architecture weights.
        """

        argmax = torch.argmax(weights)

        weighted_sum = sum(
            weights[i] * op(x, None) if i == argmax else weights[i]
            for i, op in enumerate(self.primitives)
        )

        return weighted_sum
