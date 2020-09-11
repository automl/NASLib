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
        super(MixedOp, self).__init__()
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
        super().__init__()
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







# class PCDARTSMixedOp(MixedOp):
#     """
#     Adapted from PC-DARTS code base
#     https://github.com/yuhuixu1993/PC-DARTS/blob/2f6aac375ada8aca3b9cbf782e456f8ce7e0243a/model_search.py#L25
#     """

#     def __init__(self, channel_divisor, *args, **kwargs):
#         self.channel_divisor = channel_divisor
#         kwargs['C'] = kwargs['C'] // channel_divisor
#         super(PCDARTSMixedOp, self).__init__(*args, **kwargs)
#         self.mp = nn.MaxPool2d(2, 2)

#     def forward(self, x, *args, **kwargs):
#         if 'perturb_alphas' in kwargs:
#             weights = kwargs['softmaxed_arch_weight']
#         else:
#             arch_weight = kwargs['arch_weight']
#             weights = torch.softmax(arch_weight, dim=-1)

#         dim_2 = x.shape[1]
#         xtemp = x[:, :dim_2 // self.channel_divisor, :, :]
#         xtemp2 = x[:, dim_2 // self.channel_divisor:, :, :]
#         temp1 = self.out_node_op(w * op(xtemp) for w, op in zip(weights, self._ops))
#         if temp1.shape[2] == x.shape[2]:
#             ans = torch.cat([temp1, xtemp2], dim=1)
#         else:
#             ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
#         return channel_shuffle(ans, self.channel_divisor)


# class CategoricalOp(MetaOp):
#     def __init__(self, *args, **kwargs):
#         super(CategoricalOp, self).__init__(*args, **kwargs)
#         self.ops_dict = kwargs['ops_dict']
#         self.out_node_op = eval(kwargs['out_node_op'])
#         self.build(*args, **kwargs)

#     def build(self, *args, **kwargs):
#         for primitive in self.primitives:
#             op = self.ops_dict[primitive](*args, **kwargs)
#             self._ops.append(op)

#     def forward(self, x, *args, **kwargs):
#         return self.out_node_op(op(x) for op in self._ops)
