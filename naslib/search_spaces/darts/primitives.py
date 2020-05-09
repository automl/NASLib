import torch.nn as nn

from naslib.search_spaces.core.primitives import *


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


