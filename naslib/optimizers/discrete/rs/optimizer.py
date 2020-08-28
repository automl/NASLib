import numpy as np
import torch

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.core.operations import CategoricalOp

class RandomSearch(MetaOptimizer):
    """
    Random search in DARTS is done by randomly sampling `k` architectures
    and training them for `n` epochs, then selecting the best architecture.
    DARTS paper: `k=24` and `n=100` for cifar-10.
    """

    def __init__(self, sample_size, weight_optimizer=torch.optim.SGD, loss_criteria=torch.nn.CrossEntropyLoss(), grad_clip=None):
        """
        Initialize a random search optimizer.

        Args:
            sample_size (int): Number of sampled architecures to train.
            weight_optimizer (torch.optim.Optimizer): The optimizer to 
                train the (convolutional) weights.
            loss_criteria (TODO): The loss
            grad_clip (float): Where to clip the gradients (default None).
        """
        super(RandomSearch, self).__init__()
        self.sample_size = sample_size
        self.weight_optimizer = weight_optimizer
        self.loss = loss_criteria
        self.grad_clip = grad_clip
        
        self.sampled_archs = []
        self.weight_optimizers = []

        self.validation_losses = [0 for _ in range(sample_size)]


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
        
        op_index = np.random.randint(len(current_edge_data.op))
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
            architecture_i.train()

            self.sampled_archs.append(architecture_i)
            self.weight_optimizers.append(self.weight_optimizer(architecture_i.parameters(), 0.01))


    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        self.grad_clip = 5

        for i, (arch, optim) in enumerate(zip(self.sampled_archs, self.weight_optimizers)):
            optim.zero_grad()

            # train
            logits = arch(input_train)
            loss = self.loss(logits, target_train)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(arch.parameters(), self.grad_clip)
            optim.step()
        
            # measure val loss for best architecture determination later
            self.validation_losses[i] = self.loss(arch(input_val), target_val)
        
        print('step done')


    def get_final_architecture(self):
        """
        Returns the sampled architecture with the lowest validation error.
        """
        return self.sampled_archs[np.argmin(self.validation_losses)]


    def get_op_optimizer(self):
        return self.weight_optimizer








    # @classmethod
    # def from_config(cls, *args, **kwargs):
    #     return cls(*args, **kwargs)

    # def replace_function(self, edge, graph):
    #     graph.architectural_weights = self.architectural_weights

    #     if 'op_choices' in edge:
    #         edge_key = 'cell_{}_from_{}_to_{}'.format(graph.cell_type, edge['from_node'], edge['to_node'])

    #         weights = self.architectural_weights[edge_key] if edge_key in self.architectural_weights else \
    #             torch.nn.Parameter(torch.zeros(size=[len(edge['op_choices'])],
    #                                            requires_grad=False))

    #         self.architectural_weights[edge_key] = weights
    #         edge['arch_weight'] = self.architectural_weights[edge_key]
    #         edge['op'] = CategoricalOp(primitives=edge['op_choices'], **edge['op_kwargs'])

    #     return edge

    # def uniform_sample(self, *args, **kwargs):
    #     self.set_to_zero()
    #     for arch_key, arch_weight in self.architectural_weights.items():
    #         idx = np.random.choice(len(arch_weight))
    #         arch_weight.data[idx] = 1

    # def set_to_zero(self, *args, **kwargs):
    #     for arch_key, arch_weight in self.architectural_weights.items():
    #         arch_weight.data = torch.zeros(size=[len(arch_weight)])

    # def step(self, *args, **kwargs):
    #    self.uniform_sample()

