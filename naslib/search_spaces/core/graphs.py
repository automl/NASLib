import itertools

import networkx as nx
from torch.nn import Module

from naslib.search_spaces.core.operations import MixedOp
from naslib.search_spaces.nasbench1shot1.utils import PRIMITIVES


class CellGraph(nx.DiGraph, Module):
    def __init__(self, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        Module.__init__(self)

    def forward(self, input_tensor):
        # Evaluate the graph in topological ordering
        topo_order = nx.algorithms.dag.topological_sort(self)

        # Todo: Find better way to specify the input nodes
        self.nodes[0]['output'] = input_tensor
        self.nodes[1]['output'] = input_tensor

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


class MacroGraph(nx.MultiDiGraph, Module):
    def __init__(self, *args, **kwargs):
        nx.MultiDiGraph.__init__(self, *args, **kwargs)
        Module.__init__(self)

    def forward(self, input_tensor):
        # Evaluate the graph in topological ordering
        topo_order = nx.algorithms.dag.topological_sort(self)

        # Todo: Find better way to specify the input nodes
        self.nodes[0]['output'] = input_tensor
        for node in topo_order:
            node_info = self.nodes[node]

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                edges = [self.get_edge_data(pred, node) for pred in preds]
                edges_desc = list(itertools.chain.from_iterable(edges))
                #TODO: Correct implementation of the forward pass
                '''
                for pred in preds:
                    pred_info = self.nodes[pred]
                    assert 'output' in pred_info, 'Predecessor of current node has no output.'

                    pred_output = pred_info['output']
                    op = self.get_edge_data(pred, node)['op']

                    op_outputs.append(op(pred_output))

                comb_op = node_info['comb_op']
                self.nodes[node]['output'] = comb_op(op_outputs)
                '''


if __name__ == '__main__':
    graph = CellGraph({0: {1: {'op': MixedOp(PRIMITIVES)},
                           2: {'op': MixedOp(PRIMITIVES)},
                           3: {'op': MixedOp(PRIMITIVES)}},
                       1: {2: {'op': MixedOp(PRIMITIVES)},
                           3: {'op': MixedOp(PRIMITIVES)}},
                       2: {3: {'op': MixedOp(PRIMITIVES)}}})
    graph.forward(input_tensor=None)
