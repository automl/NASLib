import torch
import torch.nn as nn
from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


# Batch Normalization from nasbench
BN_MOMENTUM = 0.997
BN_EPSILON = 1e-5

"""NASBench OPS"""

class ConvBnRelu(nn.Module):
    """
    Equivalent to conv_bn_relu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L32
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding=1):
        super(ConvBnRelu, self).__init__()
        self.op = nn.Sequential(
            # Padding = 1 is for a 3x3 kernel equivalent to tensorflow padding
            # = same
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      bias=False),
            # affine is equivalent to scale in original tensorflow code
            nn.BatchNorm2d(C_out, affine=True, momentum=BN_MOMENTUM,
                           eps=BN_EPSILON),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class Conv3x3BnRelu(nn.Module):
    """
    Equivalent to Conv3x3BnRelu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L96
    """

    def __init__(self, channels, stride):
        super(Conv3x3BnRelu, self).__init__()
        self.op = ConvBnRelu(C_in=channels, C_out=channels, kernel_size=3,
                             stride=stride)

    def forward(self, x):
        return self.op(x)


class Conv1x1BnRelu(nn.Module):
    """
    Equivalent to Conv1x1BnRelu
    https://github.com/google-research/nasbench/blob/master/nasbench/lib/base_ops.py#L107
    """

    def __init__(self, channels, stride):
        super(Conv1x1BnRelu, self).__init__()
        self.op = ConvBnRelu(C_in=channels, C_out=channels, kernel_size=1,
                             stride=stride, padding=0)

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                      bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

