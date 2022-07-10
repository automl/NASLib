import torch
from naslib.search_spaces.core.primitives import AbstractPrimitive


class Stack():
    def __init__(self):
        pass
    def __call__(self, tensors, edges_data=None):
        return torch.stack(tensors)


class Split(AbstractPrimitive):
    def __init__(self, idx):
        super().__init__(locals())
        self.idx = idx

    def forward(self, x, edge_data=None):
        return x[self.idx]