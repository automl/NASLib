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
class MetaGraph(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MetaGraph, self).__init__()

    @abstractmethod
    def _build_graph(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def parse(self, optimizer, *args, **kwargs):
        raise NotImplementedError

    def set_primitives(self, primitives):
        self.primitives = primitives

    def get_primitives(self):
        if hasattr(self, 'primitives'):
            return self.primitives
        else:
            return None

    def is_input(self, node_idx):
        return self.nodes[node_idx]['type'] == 'input'

    def is_inter(self, node_idx):
        return self.nodes[node_idx]['type'] == 'inter'

    def is_output(self, node_idx):
        return self.nodes[node_idx]['type'] == 'output'

    def input_nodes(self):
        input_nodes = [n for n in self.nodes if self.is_input(n)]
        return input_nodes

    def inter_nodes(self):
        inter_nodes = [n for n in self.nodes if self.is_inter(n)]
        return inter_nodes

    def output_nodes(self):
        output_nodes = [n for n in self.nodes if self.is_output(n)]
        return output_nodes

    @classmethod
    def from_optimizer_op(cls, optimizer, *args, **kwargs):
        graph = cls(*args, **kwargs)
        graph.parse(optimizer)
        return graph


