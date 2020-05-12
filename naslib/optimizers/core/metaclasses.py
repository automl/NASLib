from abc import ABCMeta, abstractmethod

import six
import torch.nn as nn


@six.add_metaclass(ABCMeta)
class MetaOp(nn.Module):
    def __init__(self, primitives, *args, **kwargs):
        super(MetaOp, self).__init__()
        self.primitives = primitives
        self._ops = nn.ModuleList()

    def __len__(self):
        return len(self.primitives)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build(self, C, stride, out_node_op, ops_dict):
        pass


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

    @abstractmethod
    def forward_pass_adjustment(self, *args, **kwargs):
        """
        Function evaluated prior to every forward pass
        """
        raise NotImplementedError

    @abstractmethod
    def new_epoch(self):
        """
        Function evaluated at the beginning of each new search epoch
        """
        raise NotImplementedError
