import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import math

from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.core.metaclasses import MetaOptimizer

logger = logging.getLogger(__name__)

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k=0.8):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class EdgePopUpOptimizer(MetaOptimizer):
    """
    Implementation of the DARTS paper as in
        Ramanujan et al. 2020: Edge Popup: What’s Hidden in a Randomly Weighted Neural Network?.
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
        edge.data.set("op", EdgePopUpMixedOp(primitives))

    def __init__(
        self,
        config,
        op_optimizer=torch.optim.SGD, # TODO: op_optimizer=torch.optim.Adam
        arch_optimizer=torch.optim.Adam,
        scheduler=None, #TODO: use different Cosine_Annealing
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Initialize a new instance.

        Args:

        """
        super(EdgePopUpMixedOp, self).__init__()

        self.config = config
        self.op_optimizer = op_optimizer
        self.arch_optimizer = arch_optimizer
        self.scheduler = scheduler
        self.loss = loss_criteria
        self.grad_clip = self.config.search.grad_clip

        self.architectural_weights = torch.nn.ParameterList()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.perturb_alphas = None
        self.epsilon = 0

        self.dataset = config.dataset

class EdgePopUpMixedOp(MixedOp):
    """
    Continous relaxation of the discrete search space.
    """
    def __init__(self, primitives):
        super().__init__(primitives)
        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def get_weights(self, edge_data):
        return edge_data.alpha

    def process_weights(self, weights):
        return torch.softmax(weights, dim=-1)

    def apply_weights(self, x, weights):
        subnet = GetSubnet.apply(self.scores.abs()) #TODO: remove abs()
        primitives = self.primitives * subnet
        return sum(w * op(x, None) for w, op in zip(weights, primitives))
