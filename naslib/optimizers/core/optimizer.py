from .metaclasses import MetaOptimizer
from .operations import CategoricalOp


class NASOptimizer(MetaOptimizer):
    def __init__(self, *args, **kwargs):
        super(NASOptimizer).__init__()

    def replace_function(self, edge, graph):
        if 'op_choices' in edge:
            edge['op'] = CategoricalOp(primitives=edge['op_choices'], **edge['op_kwargs'])
        return edge

    @classmethod
    def from_config(cls, *args, **kwargs):
        pass

    def forward_pass_adjustment(self, *args, **kwargs):
        """
        Function evaluated prior to every forward pass
        """
        pass

    def new_epoch(self):
        """
        Function evaluated at the beginning of each new search epoch
        """
        pass

    def perturb_alphas(self, perturbation=None, epsilon=0.3):
        pass

    def init(self, *args, **kwargs):
        pass
