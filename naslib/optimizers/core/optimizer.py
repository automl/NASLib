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

