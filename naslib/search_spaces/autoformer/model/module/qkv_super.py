from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from naslib.search_spaces.core.primitives import AbstractPrimitive


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


class Dropout_emb_choice(AbstractPrimitive):
    def __init__(self, embed_choice, super_attn_dropout, super_embed_dim):
        super(Dropout_emb_choice, self).__init__(locals())
        self.sampled_in_dim = embed_choice
        self.sample_attn_dropout = calc_dropout(super_attn_dropout,
                                                self.sampled_in_dim,
                                                super_embed_dim)

    def set_sample_config(self):
        pass

    def forward(self, x, edge_data):
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        return x

    def get_embedded_ops(self):
        return None


class QKV_super_embed_choice(AbstractPrimitive):
    def __init__(self, qkv_super, embed_choice):
        super(QKV_super_embed_choice, self).__init__(locals())
        self.qkv_super = qkv_super
        self.sampled_in_dim = embed_choice

    def set_sample_config(self):
        self.qkv_super.sample_weight = self.qkv_super.weight[:, :self.
                                                             sampled_in_dim]

    def forward(self, x, edge_data):
        self.set_sample_config()
        return x

    def get_embedded_ops(self):
        return None


class QKV_super_head_choice(AbstractPrimitive):
    def __init__(self, qkv_super, head_choice):
        super(QKV_super_head_choice, self).__init__(locals())
        self.qkv_super = qkv_super
        self.sampled_out_dim = head_choice * 64 * 3

    def set_sample_config(self):
        self.qkv_super.sample_weight = torch.cat([
            self.qkv_super.sample_weight[i:self.sampled_out_dim:3, :]
            for i in range(3)
        ],
                                                 dim=0)
        self.qkv_super.sample_bias = self.qkv_super.bias
        if self.qkv_super.bias is not None:
            self.qkv_super.sample_bias = self.qkv_super.bias[:self.
                                                             sampled_out_dim]
        self.qkv_super.sample_out_dim = self.sampled_out_dim

    def forward(self, x, edge_data):
        self.set_sample_config()
        out = self.qkv_super(x)
        print(out.shape)
        output = torch.zeros(
            [out.shape[0], out.shape[1], self.qkv_super.super_out_dim])
        output[:, :, :out.shape[-1]] = out
        #print(output.shape)
        #print(x.shape)
        return output

    def get_embedded_ops(self):
        return None


class qkv_super(nn.Linear):
    def __init__(self,
                 super_in_dim,
                 super_out_dim,
                 bias=True,
                 uniform_=None,
                 non_linear='linear',
                 scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim,
                                               self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        #self.sample_parameters()
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        #print(self.sample_weight.shape)
        #print(x.shape)
        return F.linear(
            x[:, :, :self.sample_weight.shape[-1]], self.sample_weight,
            self.sample_bias) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(self.samples['weight'].size())
        return total_flops


def sample_weight(weight, sample_in_dim, sample_out_dim):

    sample_weight = weight[:, :sample_in_dim]
    sample_weight = torch.cat(
        [sample_weight[i:sample_out_dim:3, :] for i in range(3)], dim=0)

    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias
