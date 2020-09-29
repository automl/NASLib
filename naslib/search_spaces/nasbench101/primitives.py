import torch.nn as nn

from ..core.primitives import AbstractPrimitive


"""
Code below from NASBench-01 and slighly adapted
@inproceedings{dong2020nasbench201,
  title     = {NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
"""
class ReLUConvBN(AbstractPrimitive):

    def __init__(self, C_in, C, kernel_size, stride=1, affine=False):
        super().__init__()
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C, affine=affine)
        )


    def forward(self, x, edge_data):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += '{}x{}'.format(self.kernel_size, self.kernel_size)
        return op_name


class ResNetBasicblock(AbstractPrimitive):

    def __init__(self, C_in, C_out, stride, affine=True):
        super().__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(C_in, C_out, 3, stride)
        self.conv_b = ReLUConvBN(C_out, C_out, 3)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0, bias=False))
        else:
            self.downsample = None


    def forward(self, x, edge_data):
        basicblock = self.conv_a(x, None)
        basicblock = self.conv_b(basicblock, None)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock
    

    def get_embedded_ops(self):
        return None
