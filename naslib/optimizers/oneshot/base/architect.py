from abc import abstractmethod

import numpy as np
import torch
from torch.autograd import Variable

from naslib.utils import _concat


class BaseArchitect(object):
    def __init__(self, model, momentum, weight_decay, arch_learning_rate,
                 arch_weight_decay, grad_clip=None):
        self.network_momentum = momentum
        self.network_weight_decay = weight_decay
        self.model = model
        self.grad_clip = grad_clip
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=arch_weight_decay)

    def _train_loss(self, model, input, target):
        return model._loss(input, target)

    def _val_loss(self, model, input, target):
        return model._loss(input, target)

    @abstractmethod
    def step(self, **kwargs):
        pass

    def _step(self, input_train, target_train, input_valid, target_valid, eta,
              network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train,
                                         input_valid, target_valid, eta,
                                         network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm(self.model.arch_parameters(),
                                          self.grad_clip)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        """Compute 1st order approximation"""
        loss = self._val_loss(self.model, input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid,
                                target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train,
                                                      target_train, eta,
                                                      network_optimizer)
        unrolled_loss = self._val_loss(model=unrolled_model, input=input_valid,
                                       target=target_valid)

        # Compute backwards pass with respect to the unrolled model parameters
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data if v.grad is not None else torch.zeros_like(v)
                  for v in unrolled_model.parameters()]

        # Compute expression (8) from paper
        implicit_grads = self._hessian_vector_product(vector, input_train,
                                                      target_train)

        # Compute expression (7) from paper
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self._train_loss(model=self.model, input=input, target=target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]['momentum_buffer'] for v in
                self.model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(
                loss, self.model.parameters()
            )
        ).data + self.network_weight_decay * theta

        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta)
        )
        return unrolled_model

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self._train_loss(self.model, input=input, target=target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
