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

    Attributes:
        learning_rate (float): The learning rate for optimizing operations.
        momentum (float): The momentum factor.
        weight_decay (float): Weight decay (L2 regularization).
        grad_clip (int): Gradient clipping value.
        unrolled (bool): Whether to use unrolled backpropagation or not.
        arch_learning_rate (float): The learning rate for architecture.
        arch_weight_decay (float): Weight decay for architecture.
        op_optimizer (str): Optimizer for operation weights ('SGD', 'Adam', etc.)
        arch_optimizer (str): Optimizer for architecture weights ('SGD', 'Adam', etc.)
        loss (str): Loss criterion ('CrossEntropyLoss', etc.)
        architectural_weights (torch.nn.ParameterList): List of architectural weights.
        device (torch.device): Device to run the model.
        search_space (obj): Search space for architecture.
        graph (obj): Computation graph.
        scope (str): Scope of operation.
        dataset (str): Dataset being used for search.
        arch_optimizer (obj): Torch optimizer for architecture.
        op_optimizer (obj): Torch optimizer for operations.
        loss (obj): Torch loss function.
    """

    @staticmethod
    def add_alphas(edge):
        """
        Adds architectural weights (alphas) to the edges in the computation graph.

        Args:
            edge (obj): The edge in the computation graph where alpha is to be added.

        Returns:
            None
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(
            1e-3 * torch.randn(size=[len_primitives], requires_grad=True)
        )
        edge.data.set("alpha", alpha, shared=True)

    @staticmethod
    def update_ops(edge):
        """
        Updates the operations at each edge with MixedOp.

        Args:
            edge (obj): The edge in the computation graph where operations are to be updated.

        Returns:
            None
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
        Initialize a new instance of DARTSOptimizer.

        Args:
            learning_rate (float, optional): The learning rate for optimizing operations. Defaults to 0.025.
            momentum (float, optional): The momentum factor. Defaults to 0.9.
            weight_decay (float, optional): Weight decay (L2 regularization). Defaults to 0.0003.
            grad_clip (int, optional): Gradient clipping value. Defaults to 5.
            unrolled (bool, optional): Whether to use unrolled backpropagation or not. Defaults to False.
            arch_learning_rate (float, optional): The learning rate for architecture. Defaults to 0.0003.
            arch_weight_decay (float, optional): Weight decay for architecture. Defaults to 0.001.
            op_optimizer (str, optional): Optimizer for operation weights ('SGD', 'Adam', etc.). Defaults to 'SGD'.
            arch_optimizer (str, optional): Optimizer for architecture weights ('SGD', 'Adam', etc.). Defaults to 'Adam'.
            loss_criteria (str, optional): Loss criterion ('CrossEntropyLoss', etc.). Defaults to 'CrossEntropyLoss'.
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
        """
        Adapt the search space for architecture optimization.

        Args:
            search_space (Graph): The initial search space object.
            dataset (Dataset): Dataset to be used for training/validation.
            scope (str, optional): The scope in which the graph modifications are applied. Defaults to `None`.
            **kwargs: Additional keyword arguments.

        """
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
        # logger.info("Parsed graph:\n" + graph.modules_str())

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
        """
        Get checkpointable elements of the model for saving or loading.

        Returns:
            dict: A dictionary containing all elements to be checkpointed.
        """

        return {
            "model": self.graph,
            "op_optimizer": self.op_optimizer,
            "arch_optimizer": self.arch_optimizer,
            "arch_weights": self.architectural_weights,
        }

    def before_training(self):
        """
        Prepare the model for training. This moves the graph and architectural weights to the device memory.
        """
        self.graph = self.graph.to(self.device)
        self.architectural_weights = self.architectural_weights.to(self.device)

    def new_epoch(self, epoch):
        """
        Log the architecture weights at the beginning of each new epoch.

        Args:
            epoch (int): The current epoch number.
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
        """
        Perform a single optimization step.

        Args:
            data_train (tuple): A tuple containing training input and labels.
            data_val (tuple): A tuple containing validation input and labels.

        Returns:
            tuple: A tuple containing logits for the training set, logits for the validation set,
            loss for the training set, and loss for the validation set.
        """
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
        """
        Get the final, discretized architecture based on the current architectural weights.

        Returns:
            Graph: The final architecture as a graph object.
        """
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
        """
        Get the class of the operation optimizer.

        Returns:
            type: The class type of the operation optimizer.
        """
        return self.op_optimizer.__class__

    def get_model_size(self):
        """
        Get the size of the model in terms of parameters.

        Returns:
            float: The size of the model in MB.
        """
        return count_parameters_in_MB(self.graph)

    def test_statistics(self):
        """
        Retrieve test statistics based on the current architecture and dataset.

        Returns:
            float: The test accuracy, if the graph is queryable. Otherwise, returns None.
        """
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
        """
        Perform one step of optimization for the architecture and operation weights.

        Args:
            model (torch.nn.Module): The current model architecture.
            criterion (torch.nn.Module): The loss function.
            input_train (torch.Tensor): The training input data.
            target_train (torch.Tensor): The training labels.
            input_valid (torch.Tensor): The validation input data.
            target_valid (torch.Tensor): The validation labels.
            eta (float): Learning rate.
            network_optimizer (Optimizer): The optimizer for network parameters.
            unrolled (bool): Whether to use unrolled optimization or not.

        """
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
        """
        Compute the first-order approximation of the validation loss.

        Args:
            model (torch.nn.Module): The current model architecture.
            criterion (torch.nn.Module): The loss function.
            input_valid (torch.Tensor): The validation input data.
            target_valid (torch.Tensor): The validation labels.

        """
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
        """
        Compute the loss based on the given model, criterion, inputs, and targets.

        Args:
            model (torch.nn.Module): The model architecture to use.
            criterion (torch.nn.Module): The loss function to use.
            input (torch.Tensor): The input data.
            target (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The computed loss.

        """
        pred = model(input)
        return criterion(pred, target)


class DARTSMixedOp(MixedOp):
    """
    Implements the MixedOp for DARTS, a continuous relaxation of the discrete search space.

    Attributes:
        primitives (list): List of primitive operations.

    Methods:
        get_weights: Fetches the weights associated with the edge.
        process_weights: Processes the weights for the MixedOp.
        apply_weights: Applies the processed weights to compute the MixedOp.
    """

    def __init__(self, primitives):
        """
        Initialize a DARTSMixedOp instance.

        Args:
            primitives (list): List of primitive operations.
        """
        super().__init__(primitives)

    def get_weights(self, edge_data):
        """
        Fetch the weights (alpha) associated with the edge.

        Args:
            edge_data (obj): Data associated with the edge in the graph.

        Returns:
            torch.Tensor: The weights (alpha) for this edge.
        """
        return edge_data.alpha

    def process_weights(self, weights):
        """
        Process the weights using softmax.

        Args:
            weights (torch.Tensor): The original weights.

        Returns:
            torch.Tensor: Softmax processed weights.
        """
        return torch.softmax(weights, dim=-1)

    def apply_weights(self, x, weights):
        """
        Apply the processed weights to compute the MixedOp.

        Args:
            x (torch.Tensor): Input tensor.
            weights (torch.Tensor): Processed weights.

        Returns:
            torch.Tensor: Output after applying MixedOp.
        """
        return sum(w * op(x, None) for w, op in zip(weights, self.primitives))
