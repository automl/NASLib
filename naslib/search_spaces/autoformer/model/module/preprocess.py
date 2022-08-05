
import torch
import torch.nn as nn
import torch.nn.functional as F
from naslib.search_spaces.autoformer.model.utils import trunc_normal_
from naslib.search_spaces.autoformer.model.utils import to_2tuple
import numpy as np
from naslib.search_spaces.autoformer.model.module.layernorm_super import LayerNormSuper
from naslib.search_spaces.core.primitives import AbstractPrimitive


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


class Preprocess(AbstractPrimitive):  #TODO: Better name?
    def __init__(self,
                 img_size=32,
                 patch_size=2,
                 in_chans=3,
                 embed_dim_list=[1,2,3],
                 scale=False,
                 abs_pos=True,
                 super_dropout=0.,
                 pre_norm=True):
        super(Preprocess, self).__init__(locals())

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.super_dropout = super_dropout
        self.normalize_before = pre_norm
        self.super_embed_dim = max(embed_dim_list)
        self.proj = nn.Conv2d(in_chans,
                              self.super_embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.super_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.super_embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.scale = scale

        # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None
    def sample(self, emb_choice):
        sample_embed_dim = emb_choice
        sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        sampled_bias = self.proj.bias[:sample_embed_dim, ...]
        sampled_cls_token = self.cls_token[..., :sample_embed_dim]
        sampled_pos_embed = self.pos_embed[..., :sample_embed_dim]
        sample_dropout = calc_dropout(self.super_dropout,
                                           sample_embed_dim,
                                           self.super_embed_dim)
        if self.scale:
            sampled_scale = self.super_embed_dim / sample_embed_dim
        return sampled_weight, sampled_bias, sampled_cls_token

    def forward(self, x, edge_data):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = F.conv2d(x,
                     self.sampled_weight,
                     self.sampled_bias,
                     stride=self.patch_size,
                     padding=self.proj.padding,
                     dilation=self.proj.dilation).flatten(2).transpose(1, 2)
        if self.scale:
            return x * self.sampled_scale

        x = torch.cat((self.sampled_cls_token.expand(B, -1, -1), x), dim=1)
        if self.abs_pos:
            x = x + self.sampled_pos_embed
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        output = torch.zeros([x.shape[0], x.shape[1], self.super_embed_dim])
        #print(output.shape)
        #print(x.shape)
        #print("Embedding_out_shape", x.shape)
        output[:, :, :x.shape[-1]] = x

        return output

    def get_embedded_ops(self):
        return None


class Preprocess_partial(AbstractPrimitive):
    def __init__(self, patch_emb_layer, emb_choice):
        super(Preprocess_partial, self).__init__(locals())
        self.patch_emb_layer = patch_emb_layer#.cuda()
        self.emb_choice = emb_choice

    def forward(self, x, edge_data):
        #print(x)
        sample_embed_dim = self.emb_choice
        sampled_weight = self.patch_emb_layer.proj.weight[:sample_embed_dim, ...]
        sampled_bias = self.patch_emb_layer.proj.bias[:sample_embed_dim, ...]
        sampled_cls_token = self.patch_emb_layer.cls_token[..., :sample_embed_dim]
        sampled_pos_embed = self.patch_emb_layer.pos_embed[..., :sample_embed_dim]
        sample_dropout = calc_dropout(self.patch_emb_layer.super_dropout,
                                           sample_embed_dim,
                                           self.patch_emb_layer.super_embed_dim)
        if self.patch_emb_layer.scale:
            sampled_scale = self.patch_emb_layer.super_embed_dim / sample_embed_dim
        
        #print("Patch out shape",x.shape)
        #print(x)
        B, C, H, W = x.shape
        assert H == self.patch_emb_layer.img_size[0] and W == self.patch_emb_layer.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.patch_emb_layer.img_size[0]}*{self.patch_emb_layer.img_size[1]})."
        x = F.conv2d(x,
                     sampled_weight,
                     sampled_bias,
                     stride=self.patch_emb_layer.patch_size,
                     padding=self.patch_emb_layer.proj.padding,
                     dilation=self.patch_emb_layer.proj.dilation).flatten(2).transpose(1, 2)
        if self.patch_emb_layer.scale:
            return x * sampled_scale

        x = torch.cat((sampled_cls_token.expand(B, -1, -1), x), dim=1)
        if self.patch_emb_layer.abs_pos:
            x = x + sampled_pos_embed
        x = F.dropout(x, p=sample_dropout, training=self.training)
        output = torch.zeros([x.shape[0], x.shape[1], self.patch_emb_layer.super_embed_dim],device=x.device)
        #print(output.shape)
        #print(x.shape)
        #print("Embedding_out_shape", x.shape)
        #print(edge_data)
        if not edge_data.discretize:
            output[:, :, :x.shape[-1]] = x
        else:
            output = x
        #print("output device", output.device)
        return output

    def get_embedded_ops(self):
        return None