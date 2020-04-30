import networkx as nx

from naslib.search_spaces.core.operations import MixedOp
from naslib.search_spaces.nasbench1shot1.utils import PRIMITIVES


class Graph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(*args, **kwargs)

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
        for node in topo_order:
            node_desc = graph.nodes[node]
            # Run the edges which are connected to the predecessors.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                evaled_ops = []
                for pred in preds:
                    op = self.get_edge_data(pred, node)['op']
                    evaled_ops.append(op)


if __name__ == '__main__':
    graph = Graph({0: {1: {'op': MixedOp(PRIMITIVES)},
                       2: {'op': MixedOp(PRIMITIVES)},
                       3: {'op': MixedOp(PRIMITIVES)}},
                   1: {2: {'op': MixedOp(PRIMITIVES)},
                       3: {'op': MixedOp(PRIMITIVES)}},
                   2: {3: {'op': MixedOp(PRIMITIVES)}}})
    graph.forward(input_tensor=None)
