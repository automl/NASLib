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
    def _build(self, C, stride, out_node_op, ops_dict):
        pass


@six.add_metaclass(ABCMeta)
class MetaCell(nn.Module):
    def __init__(self, graph, config):
        super(MetaCell, self).__init__()
        self.graph = graph
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


@six.add_metaclass(ABCMeta)
class MetaMacro(nn.Module):
    def __init__(self, graph, config):
        super(MetaModel, self).__init__()
        self.graph = graph
        self.config = config

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


