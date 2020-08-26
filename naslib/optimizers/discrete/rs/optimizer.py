import numpy as np
import torch

from naslib.optimizers.core import NASOptimizer
from naslib.optimizers.core.operations import CategoricalOp

class RandomSearch(NASOptimizer):
    """
    Random search in DARTS is done by randomly sampling `k` architectures
    and training them for `n` epochs, then selecting the best architecture.
    DARTS paper: `k=24` and `n=100` for cifar-10.

    TODO: This is not what is happening here, right?

    """
    def __init__(self, *args, **kwargs):
        super(RandomSearch, self).__init__()
        #self.architectural_weights = torch.nn.ParameterDict()
        self.sample_size = 5
        self.sampled_archs = []
        self.weight_optimizer = torch.optim.SGD
        self.weight_optimizers = []

    """
    These two function discretize the graph.
    """
    @staticmethod
    def add_sampled_op_index(current_edge_data):
        """
        Function to sample an op for each edge.
        """
        if current_edge_data.has('final') and current_edge_data.final:
            return current_edge_data
        
        op_index = torch.randint(len(current_edge_data.op), (1, ))
        current_edge_data.set('op_index', op_index, shared=True)
        return current_edge_data


    @staticmethod
    def update_ops(current_edge_data):
        """
        Function to replace the primitive ops at the edges
        with the sampled one
        """
        if current_edge_data.has('final') and current_edge_data.final:
            return current_edge_data
        
        primitives = current_edge_data.op
        current_edge_data.set('op', primitives[current_edge_data.op_index])
        return current_edge_data


    def adapt_search_space(self, search_space, scope=None):
        
        for i in range(self.sample_size):
            # We are going to sample several architectures
            architecture_i = search_space.clone()

            # If there is no scope defined, let's use the search space default one
            if not scope:
                scope = architecture_i.OPTIMIZER_SCOPE

            # 1. add the index first (this is shared!)
            architecture_i.update_edges(
                self.add_sampled_op_index,
                scope=scope,
                private_edge_data=False
            )

            # 2. replace primitives with respective sampled op
            architecture_i.update_edges(
                self.update_ops, 
                scope=scope,
                private_edge_data=True
            )

            architecture_i.parse()
            self.sampled_archs.append(architecture_i)
            self.weight_optimizers.append(self.weight_optimizer(architecture_i.parameters(), 0.01))
        return search_space


    def step(self, *args, **kwargs):
        graph = args[0]
        criterion = args[1]
        input_train = args[2] 
        target_train = args[3] 
        input_valid = args[4] 
        target_valid = args[5]

        grad_clip = 5#kwargs['grad_clip']

        for arch, optim in zip(self.sampled_archs, self.weight_optimizers):
            logits = arch(input_train)
            loss = criterion(logits, target_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(arch.parameters(), grad_clip)
            optim.step()
        
        
            print()


    def new_epoch(self, e):
        pass


    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def replace_function(self, edge, graph):
        graph.architectural_weights = self.architectural_weights

        if 'op_choices' in edge:
            edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

            weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
                torch.nn.Parameter(torch.zeros(size=[len(edge['op_choices'])],
                                               requires_grad=False))

            self.architectural_weights[edge_key] = weights
            edge['arch_weight'] = self.architectural_weights[edge_key]
            edge['op'] = CategoricalOp(primitives=edge['op_choices'], **edge['op_kwargs'])

        return edge

    def uniform_sample(self, *args, **kwargs):
        self.set_to_zero()
        for arch_key, arch_weight in self.architectural_weights.items():
            idx = np.random.choice(len(arch_weight))
            arch_weight.data[idx] = 1

    def set_to_zero(self, *args, **kwargs):
        for arch_key, arch_weight in self.architectural_weights.items():
            arch_weight.data = torch.zeros(size=[len(arch_weight)])

    # def step(self, *args, **kwargs):
    #    self.uniform_sample()

