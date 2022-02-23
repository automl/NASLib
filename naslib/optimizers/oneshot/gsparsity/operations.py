import torch
import torch.nn as nn
from torch.autograd import Variable

from abc import ABCMeta, abstractmethod
from naslib.search_spaces.core.primitives import AbstractPrimitive

class AvgPool1x1(AbstractPrimitive):
    """
    Implementation of Avergae Pooling with an optional
    1x1 convolution afterwards. The convolution is required
    to increase the number of channels if stride > 1.
    """

    def __init__(
        self, kernel_size, stride, C_in=None, C_out=None, affine=True, **kwargs
    ):
        super().__init__(locals())
        self.stride = stride
        self.avgpool = nn.AvgPool2d(
            3, stride=stride, padding=1, count_include_pad=False
        )
        self.weight = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.affine = affine
            self.C_in = C_in
            self.C_out = C_out
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, edge_data):
        x = self.avgpool(x)
        if self.stride > 1:
            x = self.conv(x)
            x = self.bn(x)
        else:
            x = x*self.weight
        return x

    def get_embedded_ops(self):
        return None


class Zero(AbstractPrimitive):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, stride, **kwargs):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super().__init__(locals())
        self.stride = stride
        self.weight = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x, edge_data=None):
        if self.stride == 1:
            return x.mul(0.0)*self.weight
        else:
            return self.weight*x[:, :, :: self.stride, :: self.stride].mul(0.0)

    def get_embedded_ops(self):
        return None

    def __repr__(self):
        return "Zero (stride={})".format(self.stride)


class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super().__init__(locals())
        self.weight = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x, edge_data=None):
        return x*self.weight

    def get_embedded_ops(self):
        return None