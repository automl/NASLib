import numpy as np

from naslib.optimizers.discrete.rs import RandomSearch

class RegularizedEvolution(RandomSearch):
    def __init__(self, *args, **kwargs):
        super(RegularizedEvolution, self).__init__(*args, **kwargs)

    def mutate_arch(self, parent_arch):
        self.set_to_zero()

        dim = np.random.choice(list(parent_arch))
        arch_weight = parent_arch[dim]

        argmax = int(arch_weight.argmax().data.numpy())
        list_of_idx = list(range(len(arch_weight)))
        list_of_idx.remove(argmax)
        idx = np.random.choice(list_of_idx)
        parent_arch[dim].data[argmax] = 0
        parent_arch[dim].data[idx] = 1

        self.architectural_weights = parent_arch
