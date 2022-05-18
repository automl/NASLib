import torch
import torch.nn as nn
from torch.autograd import Variable

from abc import ABCMeta, abstractmethod
import torchvision.models as models
import math


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

    forward_beforeGP = forward

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


class EdgeNormalizationCombOp(AbstractCombOp):
    """
    Combination operation to use for edge normalization.

    Returns the weighted sum of input tensors based on the (softmax of) edge weights.
    """

    def __call__(self, tensors, edges_data):
        weights = [edge_data.edge_normalization_beta for edge_data in edges_data]
        weighted_tensors = [t*w for t, w in zip(tensors, torch.softmax(torch.Tensor(weights), dim=-1))]
        return super(EdgeNormalizationCombOp, self).__call__(weighted_tensors)


class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return x

    forward_beforeGP = forward

    def get_embedded_ops(self):
        return None


class Zero(AbstractPrimitive):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, stride, C_in=None, C_out=None, **kwargs):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super().__init__(locals())
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, x, edge_data=None):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1], shape[2], shape[3] = self.C_out, (shape[2] + 1) // self.stride, (shape[3] + 1) // self.stride
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    forward_beforeGP = forward

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
            return torch.cat([x, x], dim=1)  # double the channels

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    def __init__(self, C_in=3, C_out=64, **kwargs):
        super().__init__(locals())
        self.seq = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False), nn.BatchNorm2d(C_out)
        )
    def forward(self, x, edge_data=None):
        return self.seq(x)

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    forward_beforeGP = forward

    def get_embedded_ops(self):
        return None


class ReLUConvBN(AbstractPrimitive):
    """
    Implementation of ReLU activation, followed by 2d convolution and then 2d batch normalization.
    """
    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, bias=False, track_running_stats=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x, edge_data=None):
        return self.op(x)

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    forward_beforeGP = forward

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

    forward_beforeGP = forward


class StemJigsaw(AbstractPrimitive):
    """
    This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, C_in=3, C_out=64, **kwargs):
        super().__init__(locals())
        self.seq = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, padding=1, bias=False), nn.BatchNorm2d(C_out)
        )
#         self.seq = nn.Sequential(*list(models.resnet50().children())[:-2])

    def forward(self, x, edge_data=None):
        _, _, s3, s4, s5 = x.size()
        x  = x.reshape(-1, s3, s4, s5)
        return self.seq(x)

    forward_beforeGP = forward

    def get_embedded_ops(self):
        return None


class SequentialJigsaw(AbstractPrimitive):
    """
    Implementation of `torch.nn.Sequential` to be used
    as op on edges.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(locals())
        self.primitives = args
        self.op = nn.Sequential(*args)

    def forward(self, x, edge_data):
        _, s2, s3, s4 = x.size()
        x = x.reshape(-1, 9, s2, s3, s4)
        enc_out = []
        for i in range(9):
            enc_out.append(x[:, i, :, : , :])
        x = torch.cat(enc_out, dim=1)
        return self.op(x)

    forward_beforeGP = forward

    def get_embedded_ops(self):
        return list(self.primitives)


class GenerativeDecoder(AbstractPrimitive):
    def __init__(self, in_dim, target_dim, target_num_channel=3, norm=nn.BatchNorm2d):
        super(GenerativeDecoder, self).__init__(locals())

        in_channel, in_width = in_dim[0], in_dim[1]
        out_width = target_dim[0]
        num_upsample = int(math.log2(out_width / in_width))
        assert num_upsample in [2, 3, 4, 5, 6], f'invalid num_upsample: {num_upsample}'

        self.conv1 = ConvLayer(in_channel, 1024, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv2 = ConvLayer(1024, 1024, 3, 2, 1, nn.LeakyReLU(0.2), norm)

        if num_upsample == 6:
            self.conv3 = DeconvLayer(1024, 512, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv3 = ConvLayer(1024, 512, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv4 = ConvLayer(512, 512, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        if num_upsample >= 5:
            self.conv5 = DeconvLayer(512, 256, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv5 = ConvLayer(512, 256, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv6 = ConvLayer(256, 128, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        if num_upsample >= 4:
            self.conv7 = DeconvLayer(128, 64, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv7 = ConvLayer(128, 64, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv8 = ConvLayer(64, 64, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        if num_upsample >= 3:
            self.conv9 = DeconvLayer(64, 32, 3, 2, 1, nn.LeakyReLU(0.2), norm)
        else:
            self.conv9 = ConvLayer(64, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)

        self.conv10 = ConvLayer(32, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv11 = DeconvLayer(32, 16, 3, 2, 1, nn.LeakyReLU(0.2), norm)

        self.conv12 = ConvLayer(16, 32, 3, 1, 1, nn.LeakyReLU(0.2), norm)
        self.conv13 = DeconvLayer(32, 16, 3, 2, 1, nn.LeakyReLU(0.2), norm)

        self.conv14 = ConvLayer(16, target_num_channel, 3, 1, 1, nn.Tanh(), norm)

    def forward(self, x, edge_data):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        return x

    forward_beforeGP = forward

    def get_embedded_ops(self):
        return None


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, activation, norm):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding)
        self.activation = activation
        if norm:
            if norm == nn.BatchNorm2d:
                self.norm = norm(out_channel)
            else:
                self.norm = norm
                self.conv = norm(self.conv)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm and isinstance(self.norm, nn.BatchNorm2d):
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    forward_beforeGP = forward


class DeconvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, activation, norm):
        super(DeconvLayer, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding, output_padding=1)
        self.activation = activation
        if norm == nn.BatchNorm2d:
                self.norm = norm(out_channel)
        else:
            self.norm = norm

    def forward(self, x):
        x = self.conv(x)
        if self.norm and isinstance(self.norm, nn.BatchNorm2d):
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    forward_beforeGP = forward

