import torch
import torch.nn as nn

from naslib.search_spaces.core.primitives import AbstractPrimitive


def channel_shuffle(x, groups):
    """
    https://github.com/yuhuixu1993/PC-DARTS/blob/86446d1b6bbbd5f752cc60396be13d2d5737a081/model_search.py#L9
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MixedOp(AbstractPrimitive):
    """
    Defined by the optimizer!
    """
    def __init__(self, primitives):
        super().__init__(locals())
        self.primitives = primitives
        for i, primitive in enumerate(primitives):
            self.add_module("primitive-{}".format(i), primitive)
    
    def forward(self, x, edge_data):
        normed_alphas = torch.softmax(edge_data.alpha, dim=-1)
        return sum(w * op(x, None) for w, op in zip(normed_alphas, self.primitives))
    
    def get_embedded_ops(self):
        return self.primitives


class GDASMixedOp(AbstractPrimitive):


    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed op for GDAS as defined in



        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(locals())
        self.primitives = primitives
        for i, primitive in enumerate(primitives):
            self.add_module("primitive-{}".format(i), primitive)


    def forward(self, x, edge_data):
        """
        Applies the gumbel softmax to the architecture weights
        before forwarding `x` through the graph as in DARTS
        """
        sampled_arch_weight = edge_data.sampled_arch_weight
        return sum(w * op(x, None) for w, op in zip(sampled_arch_weight, self.primitives))
    
    def get_embedded_ops(self):
        return self.primitives


