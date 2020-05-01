import torch

from naslib.search_spaces.core.graphs import Graph
from naslib.search_spaces.core.operations import TestOp


class DARTSGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ToDo: Create parser from yaml file format
        self.add_node(1, type='input', comb_op=torch.sum, preprocessing=self.preprocessing)
        self.add_node(2, type='intermediate', comb_op=torch.sum)
        self.add_node(3, type='output', comb_op=torch.cat)

        self.add_edge(1, 2, op=TestOp(), architectural_weight=None)
        self.add_edge(2, 3, op=TestOp())
        self.add_edge(1, 3, op=TestOp())

    def preprocessing(self):
        pass


if __name__ == '__main__':
    graph = DARTSGraph()
    graph.forward(input_tensor=torch.zeros(size=[1], dtype=torch.float, requires_grad=False))
    pass
