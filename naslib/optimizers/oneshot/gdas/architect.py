import torch
from torch.autograd import Variable
from torch import autograd

from naslib.utils import _concat
from naslib.optimizers.oneshot.base import BaseArchitect


class Architect(BaseArchitect):
    def __init__(self, model, momentum, weight_decay, arch_learning_rate,
                 arch_weight_decay, grad_clip=None):
        super(Architect, seld).__init__(self, model, momentum, weight_decay,
                                        arch_learning_rate, arch_weight_decay,
                                        grad_clip)


    def step(self, **kwargs):
        self._step(**kwargs)


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

        # Changes to reflect that for unused ops there will be no gradient and
        # this needs to be handled
        dtheta = _concat(
            [grad_i + self.network_weight_decay * theta_i if grad_i is not None
             else self.network_weight_decay * theta_i for grad_i, theta_i in
             zip(torch.autograd.grad(loss, self.model.parameters(),
                                     allow_unused=True),
                 self.model.parameters())]
        )

        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment
                                                                    + dtheta))
        return unrolled_model

