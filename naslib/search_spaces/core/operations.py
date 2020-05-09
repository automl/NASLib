from collections import namedtuple

from .metaclasses import MetaOp
from .primitives import *

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

OPS = {
    'noise': lambda C, stride, affine, *args, **kwargs: NoiseOp(stride, 0., 1.),
    'none': lambda C, stride, affine, *args, **kwargs: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, *args, **kwargs: nn.AvgPool2d(3, stride=stride, padding=1,
                                                                            count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, *args, **kwargs: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine, *args, **kwargs: Identity() if stride == 1 else FactorizedReduce(C, C,
                                                                                                               affine=affine),
    'conv_bn_relu_1x1': lambda C, stride, affine, *args, **kwargs: ConvBnRelu(C, C, 1, stride, 0, affine=affine),
    'conv_bn_relu_3x3': lambda C, stride, affine, *args, **kwargs: ConvBnRelu(C, C, 3, stride, 1, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, *args, **kwargs: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine, *args, **kwargs: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine, *args, **kwargs: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine, *args, **kwargs: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, *args, **kwargs: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine, *args, **kwargs: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


class TestOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.var = torch.zeros(size=[1], requires_grad=True)

    def forward(self, x):
        return x + self.var + 1


class MixedOp(MetaOp):
    def __init__(self, primitives, C, stride, out_node_op, ops_dict=OPS):
        super(MixedOp, self).__init__(primitives)
        self.build(C, stride, out_node_op, ops_dict)

    def build(self, C, stride, out_node_op='sum', ops_dict=OPS):
        self.out_node_op = eval(out_node_op)
        for primitive in self.primitives:
            op = ops_dict[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, *args, **kwargs):
        arch_weight = kwargs['arch_weight']
        weights = torch.softmax(arch_weight, dim=-1)
        return self.out_node_op(w * op(x) for w, op in zip(weights, self._ops))


class CategoricalOp(MetaOp):
    def __init__(self, *args, **kwargs):
        super(CategoricalOp, self).__init__(*args, **kwargs)
        self.ops_dict = kwargs['ops_dict']
        self.out_node_op = eval(kwargs['out_node_op'])
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        for primitive in self.primitives:
            op = self.ops_dict[primitive](*args, **kwargs)
            self._ops.append(op)

    def forward(self, x, *args, **kwargs):
        return self.out_node_op(op(x) for op in self._ops)
