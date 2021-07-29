import torch.nn as nn

from ..core.primitives import AbstractPrimitive


class ConvBNReLU(AbstractPrimitive):
    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=False):
        super().__init__(locals())
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x, edge_data):
        return self.op(x)

    def get_embedded_ops(self):
        return None


class DepthwiseConv(AbstractPrimitive):
    """
    Depthwise convolution
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(locals())
        self.op = nn.Sequential(
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x, edge_data):
        return self.op(x)

    def get_embedded_ops(self):
        return None
