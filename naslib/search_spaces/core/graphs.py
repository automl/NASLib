import networkx as nx

from naslib.search_spaces.core.metaclasses import MetaEdgeOpGraph, MetaNodeOpGraph


class EdgeOpGraph(nx.DiGraph, MetaEdgeOpGraph):
    """A graph whose edges contain operations"""

    def __init__(self, *args, **kwargs):
        nx.DiGraph.__init__(self, *args, **kwargs)
        MetaEdgeOpGraph.__init__(self)
        self._build_graph()

    def _build_graph(self):
        pass

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
        return len(self.input_nodes())

    def num_inter_nodes(self):
        return len(self.inter_nodes())

    def num_output_nodes(self):
        return len(self.output_nodes())

    def parse(self, optimizer):
        topo_order = nx.algorithms.dag.topological_sort(self)

        for node in topo_order:
            node_info = self.nodes[node]
            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                for pred in preds:
                    # Replace the operation in the edge with an optimizer compatible one.
                    edge_data = self.get_edge_data(pred, node)
                    edge_data = optimizer.replace_function(edge_data)

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
                    op = self.get_edge_data(pred, node)['op']
                    edge_output = op(pred_output)

                    edge_outputs.append(edge_output)

                # Combine evaluated input edges to form output of the cell
                comb_op = node_info['comb_op']
                self.nodes[node]['output'] = comb_op(edge_outputs)

        # Todo: Deal with multiple output EdgeOpGraphs
        return [self.nodes[node]['output'] for node in self.output_nodes()][0]


class NodeOpGraph(nx.MultiDiGraph, MetaNodeOpGraph):
    """A graph whose nodes contain operations"""

    def __init__(self, *args, **kwargs):
        nx.MultiDiGraph.__init__(self, *args, **kwargs)
        MetaNodeOpGraph.__init__(self)
        self._build_graph()

    def _build_graph(self):
        pass

    def parse(self, optimizer):
        topo_order = nx.algorithms.dag.topological_sort(self)

        for node in topo_order:
            node_info = self.nodes[node]

            # Run the edges which are connected to the current node.
            preds = list(self.predecessors(node))
            if len(preds) == 0:
                pass
            else:
                op = node_info['op']
                # Recursively run through EdgeOp graph cells
                if issubclass(type(op), EdgeOpGraph):
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
                node_info['output'] = node_info['op'](cell_input)
        return node_info['output']


if __name__ == '__main__':
    pass
