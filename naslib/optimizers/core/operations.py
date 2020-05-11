import torch
import torch.nn as nn

from .metaclasses import MetaOp


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
        arch_weight = kwargs['arch_weight']
        weights = torch.softmax(arch_weight, dim=-1)
        return self.out_node_op(w * op(x) for w, op in zip(weights, self._ops))


class GDASMixedOp(MixedOp):
    def __init__(self, *args, **kwargs):
        super(GDASMixedOp, self).__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
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
