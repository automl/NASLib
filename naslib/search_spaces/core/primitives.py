import torch
import torch.nn as nn
from torch.autograd import Variable

from abc import ABCMeta, abstractmethod


class AbstractPrimitive(nn.Module, metaclass=ABCMeta):
    """
    Use this class when creating new operations for edges.

    This is required because we are agnostic to operations
    at the edges. As a consequence, they can contain subgraphs
    which requires naslib to detect and properly process them.
    """

    def __init__(self, kwargs):
        super().__init__()

        self.init_params = {
            k: v
            for k, v in kwargs.items()
            if k != "self" and not k.startswith("_") and k != "kwargs"
        }

    @abstractmethod
    def forward(self, x, edge_data):
        """
        The forward processing of the operation.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_embedded_ops(self):
        """
        Return any embedded ops so that they can be
        analysed whether they contain a child graph, e.g.
        a 'motif' in the hierachical search space.

        If there are no embedded ops, then simply return
        `None`. Should return a list otherwise.
        """
        raise NotImplementedError()

    @property
    def get_op_name(self):
        return type(self).__name__


class AbstractCombOp(metaclass=ABCMeta):
    """
    Use this class to create custom combination operations to be used in nodes.
    """

    def __init__(self, comb_op):
        self.comb_op = comb_op

    def __call__(self, tensors, edges_data=None):
        return self.comb_op(tensors)

    @property
    def op_name(self):
        return type(self).__name__


class EdgeNormalization(AbstractCombOp):
    """
    Combination operation to use for edge normalization.

    Returns the weighted sum of input tensors based on the (softmax of) edge weights.
    """

    def __call__(self, tensors, edges_data):
        weights = [edge_data.edge_normalization_beta for edge_data in edges_data]
        weighted_tensors = [t*w for t, w in zip(tensors, torch.softmax(torch.Tensor(weights), dim=-1))]
        return super(EdgeNormalization, self).__call__(weighted_tensors)

class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
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

    def forward(self, x, edge_data=None):
        if self.stride == 1:
            return x.mul(0.0)
        else:
            return x[:, :, :: self.stride, :: self.stride].mul(0.0)

    def get_embedded_ops(self):
        return None

    def __repr__(self):
        return "Zero (stride={})".format(self.stride)


class Zero1x1(AbstractPrimitive):
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

    def forward(self, x, edge_data):
        if self.stride == 1:
            return x.mul(0.0)
        else:
            x = x[:, :, :: self.stride, :: self.stride].mul(0.0)
            return torch.cat([x, x], dim=1)  # double the channels TODO: ugly as hell

    def get_embedded_ops(self):
        return None

    def __repr__(self):
        return "Zero1x1 (stride={})".format(self.stride)


class SepConv(AbstractPrimitive):
    """
    Implementation of Separable convolution operation as
    in the DARTS paper, i.e. 2 sepconv directly after another.
    """

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, affine=True, **kwargs
    ):
        super().__init__(locals())
        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name


class DilConv(AbstractPrimitive):
    """
    Implementation of a dilated separable convolution as
    used in the DARTS paper.
    """

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, **kwargs
    ):
        super().__init__(locals())
        self.kernel_size = kernel_size
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x, edge_data):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name


class Stem(AbstractPrimitive):
    """
    This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, C_out, **kwargs):
        super().__init__(locals())
        self.seq = nn.Sequential(
            nn.Conv2d(3, C_out, 3, padding=1, bias=False), nn.BatchNorm2d(C_out)
        )

    def forward(self, x, edge_data=None):
        return self.seq(x)

    def get_embedded_ops(self):
        return None


class Sequential(AbstractPrimitive):
    """
    Implementation of `torch.nn.Sequential` to be used
    as op on edges.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(locals())
        self.primitives = args
        self.op = nn.Sequential(*args)

    def forward(self, x, edge_data):
        return self.op(x)

    def get_embedded_ops(self):
        return list(self.primitives)


class MaxPool(AbstractPrimitive):
    def __init__(self, C_in, kernel_size, stride, use_bn=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size

        if use_bn:
            self.maxpool = nn.Sequential(
                nn.MaxPool2d(kernel_size, stride=stride, padding=1),
                nn.BatchNorm2d(C_in, affine=False),
            )
        else:
            self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=1)

    def forward(self, x, edge_data):
        x = self.maxpool(x)
        return x

    def get_embedded_ops(self):
        return None


class MaxPool1x1(AbstractPrimitive):
    """
    Implementation of MaxPool with an optional 1x1 convolution
    in case stride > 1. The 1x1 convolution is required to increase
    the number of channels.
    """

    def __init__(
        self, kernel_size, stride, C_in=None, C_out=None, affine=True, **kwargs
    ):
        super().__init__(locals())
        self.stride = stride
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=1)
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, edge_data):
        x = self.maxpool(x)
        if self.stride > 1:
            x = self.conv(x)
            x = self.bn(x)
        return x

    def get_embedded_ops(self):
        return None


class AvgPool(AbstractPrimitive):
    """
    Implementation of Avergae Pooling.
    """

    def __init__(self, C_in, kernel_size, stride, use_bn=True, **kwargs):
        super().__init__(locals())

        if use_bn:
            self.avgpool = nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
                nn.BatchNorm2d(C_in, affine=False),
            )
        else:
            self.avgpool = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    def forward(self, x, edge_data):
        x = self.avgpool(x)
        return x

    def get_embedded_ops(self):
        return None


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
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, edge_data):
        x = self.avgpool(x)
        if self.stride > 1:
            x = self.conv(x)
            x = self.bn(x)
        return x

    def get_embedded_ops(self):
        return None

class GlobalAveragePooling(AbstractPrimitive):
    """
    Just a wrapper class for averaging the input across the height and width dimensions
    """

    def __init__(self):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return torch.mean(x, (2, 3))

    def get_embedded_ops(self):
        return None


class ReLUConvBN(AbstractPrimitive):
    """
    Implementation of ReLU activation, followed by 2d convolution and then 2d batch normalization.
    """
    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, bias=False, track_running_stats=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name

class ConvBnReLU(AbstractPrimitive):
    """
    Implementation of 2d convolution, followed by 2d batch normalization and ReLU activation.
    """
    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name


class InputProjection(AbstractPrimitive):
    """
    Implementation of a 1x1 projection, followed by an abstract primitive model.
    """
    def __init__(self, C_in: int, C_out: int, primitive: AbstractPrimitive):
        """
        Args:
            C_in        : Number of input channels
            C_out       : Number of output channels
            primitive   : Module of AbstractPrimitive type to which the projected input will be fed
        """
        super().__init__(locals())
        self.module = primitive
        self.op = nn.Sequential(
            ConvBnReLU(C_in, C_out, 1), # 1x1 projection
            primitive,                  # Main operation
        )

    def forward(self, x, edge_data):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"{self.module.get_op_name()}"
        return op_name


class Concat1x1(nn.Module):
    """
    Implementation of the channel-wise concatination followed by a 1x1 convolution
    to retain the channel dimension.
    """

    def __init__(self, num_in_edges, C_out, affine=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            num_in_edges * C_out, C_out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        """
        Expecting a list of input tensors. Stacking them channel-wise
        and applying 1x1 conv
        """
        x = torch.cat(x, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        return x
