from copy import deepcopy
from functools import partial
import numpy as np
import torch
import logging
from torch.autograd import Variable

from naslib.optimizers.oneshot.dartsv2.optimizer import DARTSV2Optimizer
from naslib.utils.utils import iter_flatten, AttrDict, EVLocalAvg
from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers import DARTSOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric
from numpy import linalg as LA
from torch.autograd import Variable
from analyze import Analyzer
import naslib.search_spaces.core.primitives as ops

logger = logging.getLogger(__name__)


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class RobustDARTSOptimizer(DARTSV2Optimizer):
    """
    Implementation of the DARTS paper as in
        Liu et al. 2019: DARTS: Differentiable Architecture Search.
    """

    def __init__(
            self,
            config,
            op_optimizer=torch.optim.Adam,
            arch_optimizer=torch.optim.Adam,
            loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Initialize a new instance.

        Args:

        """
        super().__init__(config, op_optimizer, arch_optimizer, loss_criteria)

        self.epochs = self.config.search.epochs
        self.early_stopping = config.search.early_stopping

        if self.early_stopping:
            self.report_freq_hessian = config.report_freq_hessian
            self.factor = config.search.factor
            self.es_start_epoch = config.search.es_start_epoch
            self.delta = config.search.delta
            self.window = config.search.window
            self.la_tracker = EVLocalAvg(self.window, self.report_freq_hessian,
                                self.epochs)

    def adapt_search_space(self, search_space, scope=None, **kwargs):
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
        if self.early_stopping:
            self.analyser = Analyzer(self.network_momentum,self.network_weight_decay,self.arch_weight_decay, self.graph)

    def step(self, data_train, data_val):

        input_train, target_train = data_train
        input_val, target_val = data_val
        eta = self.op_optimizer.state_dict()["param_groups"][0]["lr"]

        self.arch_optimizer.zero_grad()

        if self.unrolled:
            logits_val, val_loss = self.backward_step_unrolled(input_train, target_train, input_val, target_val, eta)
        else:
            logits_val, val_loss = self.backward_step(input_val, target_val)

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.architectural_weights, self.grad_clip
            )
        self.arch_optimizer.step()

        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        return logits_train, logits_val, train_loss, val_loss

    def compute_hessian(self, epoch, train_queue, valid_queue):
        if (epoch % self.report_freq_hessian == 0) or (epoch == (self.epochs - 1)):
            _data_loader = deepcopy(train_queue)
            input, target = next(iter(_data_loader))

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(non_blocking=True)

            _data_loader = deepcopy(valid_queue)
            input_search, target_search = next(iter(_data_loader))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

            # get gradient information
            # param_grads = [p.grad for p in model.parameters() if p.grad is not None]
            # param_grads = torch.cat([x.view(-1) for x in param_grads])
            # param_grads = param_grads.cpu().data.numpy()
            # grad_norm = np.linalg.norm(param_grads)

            # gradient_vector = torch.cat([x.view(-1) for x in gradient_vector])
            # grad_norm = LA.norm(gradient_vector.cpu())
            # logging.info('\nCurrent grad norm based on Train Dataset: %.4f',
            #             grad_norm)

            lr = self.op_optimizer.state_dict()["param_groups"][0]["lr"]
            H = self.analyser.compute_Hw(input, target, input_search, target_search,
                                    lr, self.op_optimizer, False)
            g = self.analyser.compute_dw(input, target, input_search, target_search,
                                    lr, self.op_optimizer, False)
            g = torch.cat([x.view(-1) for x in g])

            del _data_loader

            # early stopping
            ev = max(LA.eigvals(H.cpu().data.numpy()))

            logging.info('CURRENT EV: %f', ev)
            self.la_tracker.update(epoch, ev)

            if self.early_stop and epoch != (self.epochs - 1):
                self.la_tracker.early_stop(epoch, self.factor, self.es_start_epoch,
                                             self.delta)

    def get_checkpointables(self):
        return {
            "model": self.graph,
            "op_optimizer": self.op_optimizer,
            "arch_optimizer": self.arch_optimizer,
            "arch_weights": self.architectural_weights,
            "la_tracker": self.la_tracker
        }



