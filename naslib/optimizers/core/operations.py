import torch
import torch.nn as nn

from .metaclasses import MetaOp


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


class TestOp(MetaOp):
    def __init__(self, *args, **kwargs):
        super(TestOp).__init__(*args, **kwargs)
        self.var = torch.zeros(size=[1], requires_grad=True)

    def forward(self, x, *args, **kwargs):
        return x + self.var + 1


class MixedOp(MetaOp):
    def __init__(self, *args, **kwargs):
        super(MixedOp, self).__init__(*args, **kwargs)
        self.ops_dict = kwargs['ops_dict']
        self.out_node_op = eval(kwargs['out_node_op'])
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        for primitive in self.primitives:
            op = self.ops_dict[primitive](*args, **kwargs)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(kwargs['C'], affine=False))
            self._ops.append(op)

    def forward(self, x, *args, **kwargs):
        if 'perturb_alphas' in kwargs:
            weights = kwargs['softmaxed_arch_weight']
        else:
            arch_weight = kwargs['arch_weight']
            weights = torch.softmax(arch_weight, dim=-1)
        return self.out_node_op(w * op(x) for w, op in zip(weights, self._ops))


class GDASMixedOp(MixedOp):
    def __init__(self, *args, **kwargs):
        super(GDASMixedOp, self).__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        if 'perturb_alphas' in kwargs:
            weights = kwargs['softmaxed_arch_weight']
        else:
            weights = kwargs['sampled_arch_weight']
        cpu_weights = weights.tolist()
        use_sum = sum([abs(_) > 1e-10 for _ in cpu_weights])
        if use_sum > len(self.primitives):
            return self.out_node_op(w * op(x) for w, op in zip(weights, self._ops))
        else:
            clist = []
            for j, cpu_weight in enumerate(cpu_weights):
                if abs(cpu_weight) > 1e-10:
                    clist.append(weights[j] * self._ops[j](x))
            assert len(clist) > 0, 'invalid length : {:}'.format(cpu_weights)
            return self.out_node_op(clist)


class PCDARTSMixedOp(MixedOp):
    """
    Adapted from PC-DARTS code base
    https://github.com/yuhuixu1993/PC-DARTS/blob/2f6aac375ada8aca3b9cbf782e456f8ce7e0243a/model_search.py#L25
    """

    def __init__(self, channel_divisor, *args, **kwargs):
        self.channel_divisor = channel_divisor
        kwargs['C'] = kwargs['C'] // channel_divisor
        super(PCDARTSMixedOp, self).__init__(*args, **kwargs)
        self.mp = nn.MaxPool2d(2, 2)

    def forward(self, x, *args, **kwargs):
        if 'perturb_alphas' in kwargs:
            weights = kwargs['softmaxed_arch_weight']
        else:
            arch_weight = kwargs['arch_weight']
            weights = torch.softmax(arch_weight, dim=-1)

        dim_2 = x.shape[1]
        xtemp = x[:, :dim_2 // self.channel_divisor, :, :]
        xtemp2 = x[:, dim_2 // self.channel_divisor:, :, :]
        temp1 = self.out_node_op(w * op(xtemp) for w, op in zip(weights, self._ops))
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        return channel_shuffle(ans, self.channel_divisor)


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
