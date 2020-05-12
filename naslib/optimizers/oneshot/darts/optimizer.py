import numpy as np
import torch
from torch.autograd import Variable

from naslib.optimizers.core import NASOptimizer
from naslib.optimizers.core.operations import MixedOp
from naslib.utils import _concat


class DARTSOptimizer(NASOptimizer):
    def __init__(self, epochs, momentum, weight_decay, arch_learning_rate,
                 arch_weight_decay, grad_clip, *args, **kwargs):
        super(DARTSOptimizer, self).__init__()
        self.network_momentum = momentum
        self.network_weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.optimizer = None
        self.arch_learning_rate = arch_learning_rate
        self.arch_weight_decay = arch_weight_decay
        self.architectural_weights = torch.nn.ParameterDict()

        self.perturb_alphas = None
        self.epsilon = 0
        self.epochs = epochs
        self.edges = {}

    @classmethod
    def from_config(cls, *args, **kwargs):
        nas_opt = cls(*args, **kwargs)
        return nas_opt

    def new_epoch(self, epoch):
        if self.perturb_alphas is not None:
            self.epsilon_alpha = 0.03 + (self.epsilon - 0.03) * epoch/self.epochs

    def add_perturbation(self, perturbation=None, epsilon=.3):
        if perturbation == None:
            return
        else:
            self.perturb_alphas = perturbation


    def init(self, optimizer=torch.optim.Adam):
        self.optimizer = optimizer(
            self.architectural_weights.parameters(),
            lr=self.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=self.arch_weight_decay
        )

    def replace_function(self, edge, graph):
        graph.architectural_weights = self.architectural_weights

        if 'op_choices' in edge:
            edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

            weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
                torch.nn.Parameter(1e-3 * torch.randn(size=[len(edge['op_choices'])], requires_grad=True))

            self.architectural_weights[edge_key] = weights
            edge['arch_weight'] = self.architectural_weights[edge_key]
            edge['op'] = MixedOp(primitives=edge['op_choices'], **edge['op_kwargs'])

            if edge_key not in self.edges:
                self.edges[edge_key] = []
            self.edges[edge_key].append(edge)
        return edge

    def forward_pass_adjustment(self, *args, **kwargs):
        if self.perturb_alphas is None:
            return

        for arch_key, arch_weight in self.architectural_weights.items():
            softmaxed_arch_weight = torch.nn.functional.softmax(arch_weight.clone(),
                                                                dim=-1)
            if self.perturb_alphas == 'random':
                perturbation = torch.zeros_like(softmaxed_arch_weight).uniform_(
                    -self.epsilon_alpha, self.epsilon_alpha
                )
                softmaxed_arch_weight.data.add_(perturbation)
                # clipping
                max_index = softmaxed_arch_weight.argmax()
                softmaxed_arch_weight.data.clamp_(0, 1)
                if softmaxed_arch_weight.sum() == 0.0:
                    softmaxed_arch_weight.data[max_index] = 1.0
                softmaxed_arch_weight.data.div_(softmaxed_arch_weight.sum())

            for edge in self.edges[arch_key]:
                edge['softmaxed_arch_weight'] = softmaxed_arch_weight
                edge['perturb_alphas'] = True


    def undo_forward_pass_adjustment(self, *args, **kwargs):
        try:
            for arch_key in self.architectural_weights:
                for edge in self.edges[arch_key]:
                    del edge['softmaxed_arch_weight']
                    del edge['perturb_alphas']
        except KeyError:
            return

    def step(self, *args, **kwargs):
        self._step(*args, **kwargs)

    def _step(self, model, criterion, input_train, target_train, input_valid, target_valid, eta,
              network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(model, criterion, input_train, target_train,
                                         input_valid, target_valid, eta,
                                         network_optimizer)
        else:
            self._backward_step(model, criterion, input_valid, target_valid)

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.architectural_weights.parameters(), self.grad_clip)
        self.optimizer.step()

    def _backward_step(self, model, criterion, input_valid, target_valid):
        """Compute 1st order approximation"""
        loss = self._loss(model, criterion, input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, model, criterion, input_train, target_train, input_valid,
                                target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(model, criterion, input_train, target_train, eta,
                                                      network_optimizer)
        unrolled_loss = self._loss(model=unrolled_model, criterion=criterion, input=input_valid,
                                   target=target_valid)

        # Compute backwards pass with respect to the unrolled model parameters
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data if v.grad is not None else torch.zeros_like(v)
                  for v in unrolled_model.parameters()]

        # Compute expression (8) from paper
        implicit_grads = self._hessian_vector_product(model, criterion, vector, input_train, target_train)

        # Compute expression (7) from paper
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(self, model, criterion, input, target, eta, network_optimizer):
        loss = self._loss(model=model, criterion=criterion, input=input, target=target)
        theta = _concat(model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]['momentum_buffer'] for v in
                model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(
                loss, model.parameters()
            )
        ).data + self.network_weight_decay * theta

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
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

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
