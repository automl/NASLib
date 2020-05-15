import numpy as np
import torch

from hyperopt import tpe, hp

from naslib.optimizers.discrete.rs import RandomSearch

class TPE(RandomSearch):
    def __init__(self, *args, **kwargs):
        super(TPE, self).__init__(*args, **kwargs)
        self.architectural_weights = torch.nn.ParameterDict()
        self.space = {}
        self.fill_space()

    def step(self, *args, **kwargs):
        pass

    def fill_space(self):
        for arch_key, arch_weight in self.architectural_weights.items():
            self.space[arch_key] = hp.choice(arch_key,
                                             list(range(len(arch_weight))))
            print(list(range(len(arch_weight))))

