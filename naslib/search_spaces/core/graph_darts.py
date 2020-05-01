import torch

from naslib.search_spaces.core.graphs import Graph
from naslib.search_spaces.core.operations import TestOp


def preprocessing():
    pass


def identity(x):
    return x


class DARTSCell(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ToDo: Create parser from yaml file format
        # Input Nodes: Previous / Previous-Previous cell
        self.add_node(0, type='input', preprocessing=preprocessing, desc='previous-previous')
        self.add_node(1, type='input', preprocessing=preprocessing, desc='previous')

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


if __name__ == '__main__':
    graph = DARTSCell()
    graph.forward(input_tensor=torch.zeros(size=[1], dtype=torch.float, requires_grad=False))
    pass
