import os
import pickle
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.primitives import AbstractPrimitive
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    convert_naslib_to_str,
)
import matplotlib.pyplot as plt
from naslib.utils.utils import iter_flatten, AttrDict
from model.module.embedding_super import PatchembedSuper
from model.module.embedding_super import PatchembedSub
from naslib.utils.utils import get_project_root
from model.module.embedding_super import PatchembedSuper
from model.module.qkv_super import qkv_super
from model.module.qkv_super import QKV_super_head_choice, QKV_super_embed_choice, Dropout_emb_choice, RelativePosition2D_super, Proj_emb_choice
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper
from model.utils import trunc_normal_
from model.utils import DropPath
import numpy as np


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


#OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
class Head(AbstractPrimitive):
    def __init__(self,
                 super_in_dim,
                 super_out_dim,
                 bias=True,
                 uniform_=None,
                 non_linear='linear',
                 scale=False):
        super().__init__(locals())
        self.norm = LayerNormSuper(super_embed_dim=super_in_dim)
        self.head = LinearSuper(super_in_dim,
                                super_out_dim,
                                bias=bias,
                                uniform_=uniform_,
                                non_linear=non_linear,
                                scale=scale)

    def set_sample_config(self,
                          config,
                          sample_in_dim=None,
                          sample_out_dim=None):
        self.norm.set_sample_config(config['embed_dim'][-1])
        self.head.set_sample_config(config,
                                    sample_in_dim=sample_in_dim,
                                    sample_out_dim=sample_out_dim)

    def forward(self, x, edge_data):
        x = self.norm(x)
        return self.head(x[:, 0])

    def get_embedded_ops(self):
        return None


class PosEmbedLayer(AbstractPrimitive):
    def __init__(self, num_patches, embed_dim, abs_pos, super_dropout,
                 super_embed_dim):
        super().__init__(locals())
        self.abs_pos = abs_pos
        self.super_embed_dim = super_embed_dim
        self.super_dropout = super_dropout
        if self.abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

    def set_sample_config(self, config):
        embed_dim = config['embed_dim']
        #self.sample_mlp_ratio = self.config['mlp_ratio']
        #self.sample_layer_num = self.config['layer_num']
        #self.sample_num_heads = self.config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout, embed_dim[0],
                                           self.super_embed_dim)
        self.cls_token_sampled = self.cls_token[
            ..., :embed_dim[0]]  #.expand(B, -1, -1)
        self.pos_embed_sampled = self.pos_embed[..., :embed_dim[0]]

    def forward(self, x, edge_data):
        B = x.shape[0]

        x = torch.cat((self.cls_token_sampled.expand(B, -1, -1), x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed_sampled
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        return x

    def get_embedded_ops(self):
        return None


class TransformerEncoderLayer(AbstractPrimitive):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 pre_norm=True,
                 scale=False,
                 relative_position=False,
                 change_qkv=False,
                 max_relative_position=14):
        super().__init__(locals())

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else ops.Identity()
        self.scale = scale
        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.is_identity_layer = None
        self.attn = AttentionSuper(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=dropout,
                                   scale=self.scale,
                                   relative_position=self.relative_position,
                                   change_qkv=change_qkv,
                                   max_relative_position=max_relative_position)

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # self.dropout = dropout
        self.activation_fn = gelu
        # self.normalize_before = args.encoder_normalize_before

        self.fc1 = LinearSuper(
            super_in_dim=self.super_embed_dim,
            super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(
            super_in_dim=self.super_ffn_embed_dim_this_layer,
            super_out_dim=self.super_embed_dim)

    def set_sample_config(self, config, is_identity_layer, block_id,
                          super_dropout, super_embed_dim, super_attn_dropout):
        #self.sample_embed_dim = self.config['embed_dim']
        #self.sample_mlp_ratio = self.config['mlp_ratio']
        #self.sample_layer_num = self.config['layer_num']
        #self.sample_num_heads = self.config['num_heads']
        #self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = config['embed_dim']
        self.sample_out_dim = [
            out_dim for out_dim in self.sample_embed_dim[1:]
        ] + [self.sample_embed_dim[-1]]
        self.sample_out_dim = self.sample_out_dim[block_id]
        self.sample_mlp_ratio = config['mlp_ratio'][block_id]
        self.sample_ffn_embed_dim_this_layer = int(
            self.sample_embed_dim[block_id] * self.sample_mlp_ratio)
        self.sample_num_heads_this_layer = config['num_heads'][block_id]

        self.sample_dropout = calc_dropout(super_dropout,
                                           self.sample_embed_dim[block_id],
                                           super_embed_dim)
        self.sample_attn_dropout = calc_dropout(
            super_attn_dropout, self.sample_embed_dim[block_id],
            super_embed_dim)
        self.attn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim[block_id])

        self.attn.set_sample_config(
            sample_q_embed_dim=self.sample_num_heads_this_layer * 64,
            sample_num_heads=self.sample_num_heads_this_layer,
            sample_in_embed_dim=self.sample_embed_dim[block_id])

        self.fc1.set_sample_config(
            {},
            sample_in_dim=self.sample_embed_dim[block_id],
            sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(
            {},
            sample_in_dim=self.sample_ffn_embed_dim_this_layer,
            sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim[block_id])

    def forward(self, x, edge_data):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x

        # compute attn
        # start_time = time.time()

        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # print("attn :", time.time() - start_time)
        # compute the ffn
        # start_time = time.time()
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        # print("ffn :", time.time() - start_time)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.attn.get_complexity(sequence_length + 1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.fc1.get_complexity(sequence_length + 1)
        total_flops += self.fc2.get_complexity(sequence_length + 1)
        return total_flops

    def get_embedded_ops(self):
        return None


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


class AutoformerSearchSpace(Graph):
    """
    Implementation of the nasbench 201 search space.
    It also has an interface to the tabular benchmark of nasbench 201.
    """

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    QUERYABLE = True

    def __init__(self,
                 img_size=32,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pre_norm=True,
                 scale=False,
                 gp=False,
                 relative_position=False,
                 change_qkv=False,
                 abs_pos=True,
                 max_relative_position=14):
        super().__init__()
        self.num_classes = self.NUM_CLASSES if hasattr(self,
                                                       "NUM_CLASSES") else 10
        self.op_indices = None

        self.max_epoch = 199
        self.space_name = "autoformer"
        self.num_classes = num_classes
        self.name = "makrograph"
        # Cell is on the edges
        # 1-2:               Preprocessing
        # 2-3, ..., 6-7:     cells stage 1
        # 7-8:               residual block stride 2
        # 8-9, ..., 12-13:   cells stage 2
        # 13-14:             residual block stride 2
        # 14-15, ..., 18-19: cells stage 3
        # 19-20:             post-processing
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.block = 0
        total_num_nodes = 7
        self.add_nodes_from(range(1, 7))
        self.add_edges_from([(i, i + 1) for i in range(1, total_num_nodes)])
        self.add_edges_from([(2,7)])
        #
        self.choices = {
            'num_heads': [3, 4, 5],
            'mlp_ratio': [3.5, 4],
            'embed_dim': [1, 2, 3],
            'depth': [2,3,4]
        }
        # operations at the edges
        #
        self.depth_super = max(self.choices["depth"])
        self.super_num_heads = max(self.choices["num_heads"])
        self.super_embed_head_dim = self.super_num_heads * 64 * 3
        self.super_embed_dim = max(self.choices["embed_dim"])
        self.patch_embed_super = PatchembedSuper(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim_list=self.choices["embed_dim"],
            abs_pos=abs_pos,
            pre_norm=pre_norm)
        # preprocessing
        num_patches = self.patch_embed_super.num_patches
        self.patch_emb_op_list = []
        for e in self.choices["embed_dim"]:
            self.patch_emb_op_list.append(
                PatchembedSub(self.patch_embed_super, e))
        self.edges[1, 2].set("op", self.patch_emb_op_list)
        self.qkv_super = qkv_super(self.super_embed_dim,
                                   3 * 64 * max(self.choices["num_heads"]),
                                   bias=qkv_bias)
        self.qkv_embed_choice_list = []
        for e in self.choices["embed_dim"]:
            for h in self.choices["num_heads"]:
                self.qkv_embed_choice_list.append(
                    QKV_super_embed_choice(self.qkv_super, e, self.super_embed_dim, pre_norm))
        self.edges[2, 3].set("op", self.qkv_embed_choice_list)
        self.rel_pos_embed_k = RelativePosition2D_super(64, max_relative_position)
        self.rel_pos_embed_v = RelativePosition2D_super(64, max_relative_position)
        self.proj = LinearSuper(self.super_embed_head_dim,self.super_embed_dim)
        self.qkv_head_choice_list = []
        for h in self.choices["num_heads"]:
            self.qkv_head_choice_list.append(
                QKV_super_head_choice(self.qkv_super, self.rel_pos_embed_k, self.rel_pos_embed_v, self.proj, h, attn_drop_rate, self.super_embed_dim, self.super_embed_head_dim))
        self.edges[3, 4].set("op", self.qkv_head_choice_list)
        self.proj_emb_choice_list = []
        for e in self.choices["embed_dim"]:
            self.proj_emb_choice_list.append(
                Proj_emb_choice(self.proj, e, self.super_embed_dim))
        self.edges[4, 5].set("op", self.proj_emb_choice_list)
        self.dropout_emb_choice_list = []
        for e in self.choices["embed_dim"]:
            self.dropout_emb_choice_list.append(
                Dropout_emb_choice(e, self.super_attn_dropout,
                                   self.super_embed_dim))
        self.edges[5, 6].set("op", self.dropout_emb_choice_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth_super)]
        self.drop_path = DropPath(dpr[0]) if dpr[0] > 0. else ops.Identity()
        self.edges[6, 7].set("op", self.drop_path)
        self.edges[2, 7].set("op", ops.Identity())
    def sample_random_architecture(self, dataset_api=None):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """
        op_indices_1 = np.random.randint(3, size=(1))
        op_indices_2 = np.random.randint(3, size=(1))
        print("Choosing op 1", op_indices_1)
        print("Choosing op 2", op_indices_2)
        self.set_op_indices(op_indices_1, op_indices_2)

    def set_op_indices(self, op_indices_1, op_indices_2):
        # This will update the edges in the naslib object to op_indices
        self.edges[1, 2].set("op", self.patch_emb_op_list[op_indices_1[0]])
        self.edges[2, 3].set("op", self.qkv_embed_choice_list[op_indices_1[0]])
        self.edges[3, 4].set("op", self.qkv_head_choice_list[op_indices_2[0]])
        self.edges[4, 5].set("op", self.proj_emb_choice_list[op_indices_1[0]])
        self.edges[5, 6].set("op",
                             self.dropout_emb_choice_list[op_indices_1[0]])


def count_parameters_in_MB(model):
    return np.sum(
        np.prod(v.size()) for name, v in model.named_parameters()
        if "auxiliary" not in name) / 1e6


ss = AutoformerSearchSpace()
import networkx as nx
nx.draw(ss, with_labels = True)
plt.show()
plt.savefig('autoformer.png')
for i in range(2):
    ss.sample_random_architecture()
    #print(ss.config)
    inp = torch.randn([2, 3, 32, 32])
    out = ss(inp)
    loss = torch.sum(out)
    print(out.shape)
    #print(out)
    loss.backward()
