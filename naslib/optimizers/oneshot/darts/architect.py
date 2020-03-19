from naslib.optimizers.oneshot.base import BaseArchitect

class Architect(BaseArchitect):
    def __init__(self, model, momentum, weight_decay, arch_learning_rate,
                 arch_weight_decay, grad_clip=None):
        super(Architect, self).__init__(model, momentum, weight_decay,
                                        arch_learning_rate, arch_weight_decay,
                                        grad_clip)


    def step(self, **kwargs):
        self._step(**kwargs)

