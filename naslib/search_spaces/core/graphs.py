import itertools
import networkx as nx

from naslib.search_spaces.core.operations import MixedOp
from naslib.search_spaces.nasbench1shot1.utils import PRIMITIVES
from naslib.search_spaces.core.metaclasses import MetaEdgeOpGraph, MetaNodeOpGraph


class EdgeOpGraph(nx.DiGraph, MetaEdgeOpGraph):
    """A graph whose edges contain operations"""
    def __init__(self, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        MetaEdgeOpGraph.__init__(self)

        self.input_nodes = self.input_nodes()
        self.inter_nodes = self.inter_nodes()
        self.output_nodes = self.output_nodes()

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

    def num_input_nodes(self):
        return len(self.input_nodes)

    def num_inter_nodes(self):
        return len(self.inter_nodes)

    def num_output_nodes(self):
        return len(self.output_nodes)

    def forward(self, inputs):
        # Evaluate the graph in topological ordering
        topo_order = nx.algorithms.dag.topological_sort(self)

        input_nodes = self.get_inputs()
        assert len(input_nodes) == len(inputs), "Number of inputs isn't the same as the number of inputs in the graph"
        for input_node, input in zip(input_nodes, inputs):
            input, = input
            self.nodes[input_node]['output'] = input

        for node in topo_order:
            node_info = self.nodes[node]
            if 'preprocessing' in node_info:
                node_info['output'] = node_info['preprocessing'](node_info['output'])

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                op_outputs = []
                for pred in preds:
                    pred_info = self.nodes[pred]
                    assert 'output' in pred_info, 'Predecessor of current node has no output.'

                    pred_output = pred_info['output']
                    op = self.get_edge_data(pred, node)['op']

                    op_outputs.append(op(pred_output))

                comb_op = node_info['comb_op']
                self.nodes[node]['output'] = comb_op(op_outputs)
        return [self.nodes[node]['output'] for node in self.output_nodes()]


class NodeOpGraph(nx.MultiDiGraph, MetaNodeOpGraph):
    """A graph whose nodes contain operations"""
    def __init__(self, *args, **kwargs):
        nx.MultiDiGraph.__init__(self, *args, **kwargs)
        MetaNodeOpGraph.__init__(self)

    def forward(self, *inputs):
        # Evaluate the graph in topological ordering
        topo_order = nx.algorithms.dag.topological_sort(self)

        # Todo: Find better way to specify the input nodes
        self.nodes[0]['output'] = inputs
        for node in topo_order:
            node_info = self.nodes[node]

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                cell_input = [self.nodes[pred]['output'] for pred in preds]
                node_info['output'] = node_info['op'](*cell_input)


if __name__ == '__main__':
    graph = EdgeOpGraph({0: {1: {'op': MixedOp(PRIMITIVES)},
                             2: {'op': MixedOp(PRIMITIVES)},
                             3: {'op': MixedOp(PRIMITIVES)}},
                         1: {2: {'op': MixedOp(PRIMITIVES)},
                             3: {'op': MixedOp(PRIMITIVES)}},
                         2: {3: {'op': MixedOp(PRIMITIVES)}}})
    graph.forward(input_tensor=None)
