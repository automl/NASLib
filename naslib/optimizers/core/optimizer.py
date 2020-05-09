import six
from abc import ABCMeta, abstractmethod

from naslib.search_spaces.core.operations import CategoricalOp

@six.add_metaclass(ABCMeta)
class MetaOptimizer(object):
    def __init__(self):
        super(MetaOptimizer, self).__init__()

    @abstractmethod
    def replace_function(self, edge, graph):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        pass


class NASOptimizer(MetaOptimizer):
    def __init__(self):
        super(NASOptimizer).__init__()

    def replace_function(self, edge, graph):
        if 'op_choices' in edge:
            edge['op'] = CategoricalOp(primitives=edge['op_choices'], **edge['op_kwargs'])
        return edge

    @classmethod
    def from_config(cls, *args, **kwargs):
        pass
