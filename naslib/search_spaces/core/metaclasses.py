import six
import torch.nn as nn
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class MetaOp(nn.Module):
    def __init__(self, primitives):
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
class MetaEdgeOpGraph(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MetaEdgeOpGraph, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


@six.add_metaclass(ABCMeta)
class MetaNodeOpGraph(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MetaNodeOpGraph, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


