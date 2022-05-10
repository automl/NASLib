import torch
import torch.nn as nn

# OPS defines operations for micro cell structures
OPS = {
    '0': lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
    '1': lambda C_in, C_out, stride, affine, track_running_stats: Identity() if (
                stride == 1 and C_in == C_out) else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
    '2': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1, 1), stride, (0, 0),
                                                                             (1, 1), affine, track_running_stats),
    '3': lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3, 3), stride, (1, 1),
                                                                             (1, 1), affine, track_running_stats)
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats,
                 activation='relu'):
        super(ReLUConvBN, self).__init__()
        if activation == 'leaky':
            ops = [nn.LeakyReLU(0.2, False)]
        elif activation == 'relu':
            ops = [nn.ReLU(inplace=False)]
        else:
            raise ValueError(f"invalid activation {activation}")
        ops += [nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)]
        self.ops = nn.Sequential(*ops)
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x):
        return self.ops(x)

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1], shape[2], shape[3] = self.C_out, (shape[2] + 1) // self.stride, (shape[3] + 1) // self.stride
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
        C_outs = [C_out // 2, C_out - C_out // 2]
        self.convs = nn.ModuleList()
        for i in range(2):
            self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        # print(self.convs[0](x).shape, self.convs[1](y[:,:,1:,1:]).shape)
        # print(out.shape)

        out = self.bn(out)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


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


class DeconvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, activation, norm):
        super(DeconvLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride=stride, padding=padding,
                                       output_padding=1)
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
