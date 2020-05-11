import networkx as nx

from naslib.utils.utils import drop_path, cat_channels
from .metaclasses import MetaGraph
from .primitives import Identity

# Todo: Remove 'eval' functionality with something safer
test = cat_channels


class EdgeOpGraph(nx.DiGraph, MetaGraph):
    """A graph whose edges contain operations"""

    def __init__(self, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        MetaGraph.__init__(self)
        self._build_graph()

    def _build_graph(self):
        pass

    def save_graph(self):
        pass

    def parse(self, optimizer, *args, **kwargs):
        topo_order = nx.algorithms.dag.topological_sort(self)

        for node in topo_order:
            node_info = self.nodes[node]
            if 'preprocessing' in node_info:
                self.add_module('node' + str(node), node_info['preprocessing'])

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                for pred in preds:
                    # Replace the operation in the edge with an optimizer compatible one.
                    edge_data = self.get_edge_data(pred, node)
                    edge_data = optimizer.replace_function(edge_data, self)
                    self.add_module('edge(%d,%d)' % (pred, node), edge_data['op'])

    def forward(self, inputs):
        # Evaluate the graph in topological ordering
        topo_order = nx.algorithms.dag.topological_sort(self)

        # Todo deal with multidigraph input.
        input_nodes = self.input_nodes()
        assert len(input_nodes) == len(inputs), "Number of inputs isn't the same as the number of inputs in the graph"

        for input_node, input in zip(input_nodes, inputs):
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
                edge_outputs = []
                for pred in preds:
                    pred_info = self.nodes[pred]
                    assert 'output' in pred_info, 'Predecessor of current node has no output.'

                    # Evaluate the edge from the predecessor to the current node.
                    pred_output = pred_info['output']
                    edge_data = self.get_edge_data(pred, node)
                    edge_output = edge_data['op'](pred_output, **edge_data)

                    if hasattr(self, 'drop_path_prob'):
                        if self.training and self.drop_path_prob > 0:
                            edge_op = self.get_edge_op(pred, node)
                            if self.is_inter(node):
                                if len(edge_op) == 1:
                                    if not isinstance(edge_op._ops[0], Identity):
                                        edge_output = drop_path(edge_output,
                                                                self.drop_path_prob)
                                else:
                                    edge_output = drop_path(edge_output,
                                                            self.drop_path_prob)
                    edge_outputs.append(edge_output)

                # Combine evaluated input edges to form output of the cell
                comb_op = eval(node_info['comb_op'])
                self.nodes[node]['output'] = comb_op(edge_outputs)

        # Todo: Deal with multiple output EdgeOpGraphs
        return [self.nodes[node]['output'] for node in self.output_nodes()][0]


class NodeOpGraph(nx.DiGraph, MetaGraph):
    """A graph whose nodes contain operations"""

    def __init__(self, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        MetaGraph.__init__(self)
        self._build_graph()

    def _build_graph(self):
        pass

    def save_graph(self, filename='graph.yaml', save_arch_weights=False):
        MetaGraph.save_graph(self, filename, save_arch_weights)

    def parse(self, optimizer, *args, **kwargs):
        topo_order = nx.algorithms.dag.topological_sort(self)

        for node in topo_order:
            node_info = self.nodes[node]
            if 'op' in node_info:
                self.add_module('node' + str(node), node_info['op'])

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                op = node_info['op']
                # Recursively run through EdgeOp graph cells
                if issubclass(type(op), EdgeOpGraph):
                    self.add_module('node' + str(node), op)
                    op.parse(optimizer)

    def forward(self, inputs):
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
                if 'transform' in node_info:
                    cell_input = node_info['transform'](cell_input)
                node_info['output'] = node_info['op'](cell_input)
        return [self.nodes[node]['output'] for node in self.output_nodes()][0]


if __name__ == '__main__':
    pass
