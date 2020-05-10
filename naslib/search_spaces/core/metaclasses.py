from abc import ABCMeta, abstractmethod

import six
import yaml
import torch.nn as nn

from naslib.utils import exception


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
        return self.get_node_type(node_idx) == 'input'

    def is_inter(self, node_idx):
        return self.get_node_type(node_idx) == 'inter'

    def is_output(self, node_idx):
        return self.get_node_type(node_idx) == 'output'

    def input_nodes(self):
        input_nodes = [n for n in self.nodes if self.is_input(n)]
        return input_nodes

    def inter_nodes(self):
        inter_nodes = [n for n in self.nodes if self.is_inter(n)]
        return inter_nodes

    def output_nodes(self):
        output_nodes = [n for n in self.nodes if self.is_output(n)]
        return output_nodes

    def get_node_attributes(self, node_idx, exclude=None):
        node_attr_dict = self.nodes[node_idx]
        if exclude is not None:
            node_attr_dict = {k: v for k, v in node_attr_dict.items() if k not in exclude}
        return node_attr_dict

    def get_edge_attributes(self, from_node, to_node, exclude=None):
        edge_attr_dict = self[from_node][to_node]
        if exclude is not None:
            edge_attr_dict = {k: v for k, v in edge_attr_dict.items() if k not in exclude}
        return edge_attr_dict

    @exception(KeyError)
    def get_node_preprocessing(self, node_idx):
        return self.nodes[node_idx]['preprocessing']

    @exception(KeyError)
    def get_node_op(self, node_idx):
        return self.nodes[node_idx]['op']

    @exception(KeyError)
    def get_node_type(self, node_idx):
        return self.nodes[node_idx]['type']

    @exception(KeyError)
    def get_edge_op(self, from_node, to_node):
        return self[from_node][to_node]['op']

    @exception(KeyError)
    def get_edge_op_choices(self, from_node, to_node):
        return self[from_node][to_node]['op_choices']

    @exception(KeyError)
    def get_edge_op_kwargs(self, from_node, to_node):
        return self[from_node][to_node]['op_kwargs']

    @exception(KeyError)
    def get_edge_arch_weights(self, from_node, to_node):
        return self[from_node][to_node]['arch_weight']

    @classmethod
    def from_optimizer_op(cls, optimizer, *args, **kwargs):
        graph = cls(*args, **kwargs)
        graph.parse(optimizer)
        return graph

    @classmethod
    def from_config(cls, *args, **kwargs):
        return NotImplementedError

    @staticmethod
    def save_graph(graph, filename=None, save_arch_weights=False):
        _graph = {'type': None, 'nodes': {}, 'edges': {}}
        _graph.update({'type': type(graph).__name__})
        if hasattr(graph, 'primitives'):
            _graph['primitives'] = str(graph.primitives)

        # exctract node attributes and add them to dict
        for node in graph.nodes:
            node_attributes = graph.get_node_attributes(node,
                                                        exclude=['output',
                                                                 'preprocessing',
                                                                 'transform',
                                                                 'op'])
            _graph['nodes'].update({node: {k: str(v) for k, v in
                                           node_attributes.items()}})
            if graph.get_node_preprocessing(node) is not None:
                _graph['nodes'][node].update({'preprocessing':
                                                  type(graph.get_node_preprocessing(node)).__name__})

            if hasattr(graph.get_node_op(node), 'save_graph'):
                _graph['nodes'][node].update({'op':
                                                  MetaGraph.save_graph(graph.get_node_op(node),
                                                                       filename=None,
                                                                       save_arch_weights=save_arch_weights)})
            elif graph.get_node_op(node) is not None:
                _graph['nodes'][node].update({'op':
                                                  type(graph.get_node_op(node)).__name__})

        # exctract edge attributes and add them to dict
        for edge in graph.edges:
            exclude_list = ['op']
            if not save_arch_weights:
                exclude_list.append('arch_weight')
            edge_attributes = graph.get_edge_attributes(*edge,
                                                        exclude=exclude_list)
            _graph['edges'].update({str(edge): {k: str(v) for k, v in
                                                edge_attributes.items()}})
            if graph.get_edge_op(*edge) is not None:
                _graph['edges'][str(edge)].update({'op':
                                                       type(graph.get_edge_op(*edge)).__name__})

        if filename is None:
            return _graph
        else:
            with open(filename, 'w') as f:
                yaml.safe_dump(_graph, f)
