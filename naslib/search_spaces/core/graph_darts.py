import numpy as np
import torch

from naslib.search_spaces.core.graphs import CellGraph, MacroGraph
from naslib.search_spaces.core.operations import TestOp


def preprocessing():
    pass


def identity(x):
    return x


class DARTSCell(CellGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ToDo: Create parser from yaml file format
        # Input Nodes: Previous / Previous-Previous cell
        self.add_node(0, type='input', preprocessing=identity, desc='previous-previous')
        self.add_node(1, type='input', preprocessing=identity, desc='previous')

        # 4 intermediate nodes
        self.add_node(2, type='inter', comb_op=sum)
        self.add_node(3, type='inter', comb_op=sum)
        self.add_node(4, type='inter', comb_op=sum)
        self.add_node(5, type='inter', comb_op=sum)

        # Output node
        self.add_node(6, type='output', comb_op=torch.cat)

        # Edges: input-inter and inter-inter
        for to_node in range(2, 6):
            for from_node in range(to_node):
                self.add_edge(from_node, to_node, op=TestOp())

        # Edges: inter-output
        self.add_edge(2, 6, op=identity)
        self.add_edge(3, 6, op=identity)
        self.add_edge(4, 6, op=identity)
        self.add_edge(5, 6, op=identity)


class DARTSMacroGraph(MacroGraph):
    def __init__(self, num_cells=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_reduction = 2
        num_normal = num_cells - num_reduction
        self.add_node(0, type='input')
        self.add_node(1, op=identity, type='stem')

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

        # Parallel edge that's why MultiDiGraph
        self.add_edge(1, 2, type='input', desc='previous-previous')
        self.add_edge(1, 2, type='input', desc='previous')

        for i in range(3, num_cells + 2):
            self.add_edge(i - 2, i, type='input', desc='previous-previous')
            self.add_edge(i - 1, i, type='input', desc='previous')

        # From output of normal-reduction cell to pooling layer
        self.add_edge(num_cells + 1, num_cells + 2)
        self.add_edge(num_cells + 2, num_cells + 3)


if __name__ == '__main__':
    graph = DARTSMacroGraph()
    graph(input_tensor=torch.zeros(size=[1], dtype=torch.float, requires_grad=False))
    pass
