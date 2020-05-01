import networkx as nx
from torch.nn import Module

from naslib.search_spaces.core.operations import MixedOp
from naslib.search_spaces.nasbench1shot1.utils import PRIMITIVES


class Graph(nx.DiGraph, Module):
    def __init__(self, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        Module.__init__(self)
        # if len(self) <= 2:
        #    raise('The graph needs at least 1 intermediate node')
        self._init_node_attributes()
        self.input_nodes = self.input_nodes()
        self.op_nodes = None
        self.intermediate_nodes = self.intermediate_nodes()
        self.output_nodes = self.output_nodes()

        self._primitives = self.get_op_choices()

    def _init_node_attributes(self):
        for node in self.nodes:
            self.nodes[node].update(
                {k: v for k, v in {
                    'input': False, 'inter': False, 'output': False,
                    'operator': 'sum'
                }.items() if k not in self.nodes[node]}
            )

    def get_op_choices(self):
        for i, j in self.edges:
            if hasattr(self[i][j]['op'], 'primitives'):
                return self[i][j]['op'].primitives

    @classmethod
    def from_dict(cls, graph_dict: dict):
        pass

    def is_input(self, node_idx):
        return self.nodes[node_idx]['input']

    def is_intermediate(self, node_idx):
        return self.nodes[node_idx]['inter']

    def is_output(self, node_idx):
        return self.nodes[node_idx]['output']

    def input_nodes(self):
        input_nodes = [n for n in self.nodes if self.is_input(n)]
        return input_nodes

    def intermediate_nodes(self):
        inter_nodes = [n for n in self.nodes if self.is_intermediate(n)]
        return inter_nodes

    def output_nodes(self):
        output_nodes = [n for n in self.nodes if self.is_output(n)]
        return output_nodes

    def num_input_nodes(self):
        return len(self.input_nodes)

    def num_intermediate_nodes(self):
        return len(self.intermediate_nodes)

    def num_output_nodes(self):
        return len(self.output_nodes)

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


if __name__ == '__main__':
    graph = Graph({0: {1: {'op': MixedOp(PRIMITIVES)},
                       2: {'op': MixedOp(PRIMITIVES)},
                       3: {'op': MixedOp(PRIMITIVES)}},
                   1: {2: {'op': MixedOp(PRIMITIVES)},
                       3: {'op': MixedOp(PRIMITIVES)}},
                   2: {3: {'op': MixedOp(PRIMITIVES)}}})
    graph.forward(input_tensor=None)
