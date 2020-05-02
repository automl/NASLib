import numpy as np
import torch

from naslib.search_spaces.core.graphs import EdgeOpGraph, NodeOpGraph
from naslib.search_spaces.core.primitives import Identity, FactorizedReduce, ReLUConvBN


class DARTSCell(EdgeOpGraph):
    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ToDo: Create parser from yaml file format
        # Input Nodes: Previous / Previous-Previous cell
        self.add_node(0, type='input', preprocessing=Identity(), desc='previous-previous')
        self.add_node(1, type='input', preprocessing=Identity(), desc='previous')

        # 4 intermediate nodes
        self.add_node(2, type='inter', comb_op=sum)
        self.add_node(3, type='inter', comb_op=sum)
        self.add_node(4, type='inter', comb_op=sum)
        self.add_node(5, type='inter', comb_op=sum)

        # Output node
        self.add_node(6, type='output', comb_op=torch.cat)

        # Edges: input-inter and inter-inter
        for to_node in self.inter_nodes():
            for from_node in range(to_node):
                stride = 2 if self.graph['type'] == 'reduction' and from_node < 2 else 1
                self.add_edge(
                    from_node, to_node, op=CategoricalOp(
                        primitives=PRIMITIVES, C=C, stride=stride,
                        out_node_op=to_node['comb_op']
                    )
                )

        # Edges: inter-output
        self.add_edge(2, 6, op=Identity())
        self.add_edge(3, 6, op=Identity())
        self.add_edge(4, 6, op=Identity())
        self.add_edge(5, 6, op=Identity())


class DARTSMacroGraph(NodeOpGraph):
    def __init__(self, num_cells=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_node(0, type='input')
        self.add_node(1, op=identity, type='stem')
        self.add_node('1b', op=identity, type='stem')

        # Normal and reduction cells
        # Todo: Add channel count computation
        for cell_num in range(2, num_cells + 2):
            if cell_num == np.floor(num_cells / 3) or cell_num == np.floor(2 * num_cells / 3):
                self.add_node(cell_num, op=DARTSCell(type='reduction'), type='reduction')
            else:
                self.add_node(cell_num, op=DARTSCell(type='normal'), type='normal')

        self.add_node(num_cells + 2, op=identity, type='pooling')
        self.add_node(num_cells + 3, op=identity, type='classification')

        # Edges
        self.add_edge(0, 1)
        self.add_edge(0, '1b')

        # Parallel edge that's why MultiDiGraph
        self.add_edge(1, 2, type='input', desc='previous-previous')
        self.add_edge('1b', 2, type='input', desc='previous')

        for i in range(3, num_cells + 2):
            self.add_edge(i - 2, i, type='input', desc='previous-previous')
            self.add_edge(i - 1, i, type='input', desc='previous')

        # From output of normal-reduction cell to pooling layer
        self.add_edge(num_cells + 1, num_cells + 2)
        self.add_edge(num_cells + 2, num_cells + 3)


if __name__ == '__main__':
    # graph = DARTSMacroGraph()
    # graph(*torch.zeros(size=[1], dtype=torch.float, requires_grad=False))
    graph = DARTSCell()
    graph([torch.zeros(size=[1], dtype=torch.float, requires_grad=False) for _ in range(2)])
    pass
