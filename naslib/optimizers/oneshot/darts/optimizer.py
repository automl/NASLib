import numpy as np
import torch
import logging
from torch.autograd import Variable

from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.utils.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

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
        edge.data.set("op", MixedOp(primitives))

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
        super(DARTSOptimizer, self).__init__()

        self.config = config
        self.op_optimizer = op_optimizer
        self.arch_optimizer = arch_optimizer
        self.loss = loss_criteria
        self.grad_clip = self.config.search.grad_clip

        self.architectural_weights = torch.nn.ParameterList()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.perturb_alphas = None
        self.epsilon = 0

        self.dataset = config.dataset

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
        #logger.info("Parsed graph:\n" + graph.modules_str())

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
        unrolled_model = self._compute_unrolled_model(
            model, criterion, input_train, target_train, eta, network_optimizer
        )
        unrolled_loss = self._loss(
            model=unrolled_model,
            criterion=criterion,
            input=input_valid,
            target=target_valid,
        )

        # Compute backwards pass with respect to the unrolled model parameters
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [
            v.grad.data if v.grad is not None else torch.zeros_like(v)
            for v in unrolled_model.parameters()
        ]

        # Compute expression (8) from paper
        implicit_grads = self._hessian_vector_product(
            model, criterion, vector, input_train, target_train
        )

        # Compute expression (7) from paper
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(
        self, model, criterion, input, target, eta, network_optimizer
    ):
        loss = self._loss(model=model, criterion=criterion, input=input, target=target)
        theta = _concat(model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]["momentum_buffer"]
                for v in model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = (
            _concat(torch.autograd.grad(loss, model.parameters())).data
            + self.network_weight_decay * theta
        )

        unrolled_model = self._construct_model_from_theta(
            model, theta.sub(eta, moment + dtheta)
        )
        return unrolled_model

    def _construct_model_from_theta(self, model, theta):
        model_new = model.new()
        model_dict = model.state_dict()

        params, offset = {}, 0
        for k, v in model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset : offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        model_new = model_new.to(self.device)

        return model_new

    def _hessian_vector_product(self, model, criterion, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)
        train_loss = self._loss(model, criterion, input=input, target=target)
        grads_p = torch.autograd.grad(train_loss, model.arch_parameters())

        for p, v in zip(model.parameters(), vector):
            p.data.sub_(2 * R, v)
        train_loss = self._loss(model, criterion, input=input, target=target)
        grads_n = torch.autograd.grad(train_loss, model.arch_parameters())

        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def _loss(self, model, criterion, input, target):
        pred = model(input)
        return criterion(pred, target)


class MixedOp(AbstractPrimitive):
    """
    Continous relaxation of the discrete search space.
    """

    def __init__(self, primitives):
        super().__init__(locals())
        self.primitives = primitives
        self.add_primitive_modules()

    def add_primitive_modules(self):
        for i, primitive in enumerate(self.primitives):
            self.add_module("primitive-{}".format(i), primitive)

    def forward(self, x, edge_data):
        normed_alphas = torch.softmax(edge_data.alpha, dim=-1)
        return sum(w * op(x, None) for w, op in zip(normed_alphas, self.primitives))

    def get_embedded_ops(self):
        return self.primitives

    def set_embedded_ops(self, primitives):
        self.primitives = primitives

class PartialConnectionOp(AbstractPrimitive):
    """
    Partial Connection Operation.

    This class takes a MixedOp and replaces its primitives with the fewer channel version of those primitives.
    """

    def __init__(self, mixed_op: MixedOp, k: int):
        super().__init__(locals())
        self.k = k
        self.mixed_op = mixed_op

        pc_primitives = []
        for primitive in mixed_op.get_embedded_ops():
            pc_primitives.append(self._create_pc_primitive(primitive))

        self.mixed_op.set_embedded_ops(pc_primitives)
        self.mixed_op.add_primitive_modules()

    def _create_pc_primitive(self, primitive: AbstractPrimitive) -> AbstractPrimitive:
        """
        Creates primitives with fewer channels for Partial Connection operation.
        """
        init_params = primitive.init_params
        self.mp = torch.nn.MaxPool2d(2,2)

        try:
            #TODO: Force all AbstractPrimitives with convolutions to use 'C_in' and 'C_out' in the initializer
            init_params['C_in'] = init_params['C_in']//self.k

            if 'C_out' in init_params:
                init_params['C_out'] = init_params['C_out']//self.k
            elif 'C' in init_params:
                init_params['C'] = init_params['C']//self.k
        except KeyError:
            return primitive

        pc_primitive = primitive.__class__(**init_params)
        return pc_primitive

    def _shuffle_channels(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.k

        # reshape
        x = x.view(batchsize, self.k, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, x, edge_data):
        dim_2 = x.shape[1]
        xtemp = x[ : , :  dim_2//self.k, :, :]
        xtemp2 = x[ : ,  dim_2//self.k:, :, :]

        temp1 = self.mixed_op(xtemp, edge_data)

        if temp1.shape[2] == x.shape[2]:
            result = torch.cat([temp1, xtemp2],dim=1)
        else:
            # TODO: Verify that downsampling in every graph reduces the size in exactly half
            result = torch.cat([temp1, self.mp(xtemp2)], dim=1)

        result = self._shuffle_channels(result)
        return result

    def get_embedded_ops(self):
        return self.mixed_op.get_embedded_ops()