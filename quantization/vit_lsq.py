##############################################
# ViT‑LSQ: Vision Transformer with LSQ blocks
# ‑ Converted from vit_quant.py (PTQ) to LSQ
# ‑ Uses lsq_layer.QuantLinear / QuantConv2d / QuantAct
# ‑ No running_stat; model_freeze / model_unfreeze removed
##############################################

import os, math, re, warnings
from itertools import repeat
import collections.abc
from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from .lsq_layer import QuantLinear, QuantConv2d, QuantAct
from .layers_quant import PatchEmbed, HybridEmbed, Mlp, DropPath, trunc_normal_

__all__ = ["deit_tiny_patch16_224_lsq", "deit_small_patch16_224_lsq", "deit_base_patch16_224_lsq"]

########################################
# Attention / Block remain identical
#   except: they already rely on Quant*
#   which are now LSQ implementations.
########################################

def find_non_zero_patches(images, patch_size):
    bs, c, h, w = images.shape
    patch_h, patch_w = patch_size, patch_size
    if h % patch_h != 0 or w % patch_w != 0:
        raise ValueError("Image dimensions are not divisible by patch size")

    images_reshaped = images.reshape(bs, c, h // patch_h, patch_h, w // patch_w, patch_w)

    images_transposed = images_reshaped.permute(0, 2, 4, 1, 3, 5)

    images_patches = images_transposed.reshape(bs, -1, c * patch_h * patch_w)

    non_zero_patches = torch.any(images_patches != 0, dim=2)
    non_zero_indices = []
    for i in range(bs):
        a = torch.nonzero(non_zero_patches[i], as_tuple=False).squeeze() + 1

        # print(a.size())
        non_zero_indices.append(a)

    # non_zero_indices = [torch.nonzero(non_zero_patches[i], as_tuple=False).squeeze() + 1 for i in range(bs)]
    non_zero_indices=torch.stack(non_zero_indices,dim=0)
    return non_zero_indices

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., cfg=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = QuantLinear(dim, dim * 3, nbits=cfg.weight_bit)
        self.qact1 = QuantAct(nbits=cfg.activation_bit)
        self.qact2 = QuantAct(nbits=cfg.activation_bit)
        self.proj = QuantLinear(dim, dim, nbits=cfg.weight_bit)
        self.qact3 = QuantAct(nbits=cfg.activation_bit)
        self.qact_attn1 = QuantAct(nbits=cfg.activation_bit)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        x = self.qkv(x)
        x = self.qact1(x)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.qact_attn1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.qact2(x)
        x = self.proj(x)
        x = self.qact3(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.qact1 = QuantAct(nbits=cfg.activation_bit)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, cfg=cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.qact2 = QuantAct(nbits=cfg.activation_bit)
        self.norm2 = norm_layer(dim)
        self.qact3 = QuantAct(nbits=cfg.activation_bit)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, cfg=cfg)
        self.qact4 = QuantAct(nbits=cfg.activation_bit)

    def forward(self, x):
        x = self.qact2(x + self.drop_path(self.attn(self.qact1(self.norm1(x)))))
        x = self.qact4(x + self.drop_path(self.mlp(self.qact3(self.norm2(x)))))
        return x

########################################
# VisionTransformer – main changes:
#   • removed model_freeze/unfreeze
#   • model_quant() no longer toggles
#     running_stat, just placeholder
########################################

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, input_quant=False, cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.cfg = cfg

        self.input_quant = input_quant
        if input_quant:
            self.qact_input = QuantAct(nbits=cfg.activation_bit)

        # patch embed
        if hybrid_backbone is None:
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, cfg=cfg)
        else:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # LSQ activations
        self.qact_embed = QuantAct(nbits=cfg.activation_bit)
        self.qact_pos = QuantAct(nbits=cfg.activation_bit)
        self.qact1 = QuantAct(nbits=cfg.activation_bit)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                  attn_drop_rate, dpr[i], norm_layer=norm_layer, cfg=cfg)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.qact2 = QuantAct(nbits=cfg.activation_bit)

        self.pre_logits = nn.Identity() if representation_size is None else nn.Sequential(
            OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh()),
            ])
        )
        if representation_size:
            self.num_features = representation_size

        self.head = QuantLinear(self.num_features, num_classes, nbits=cfg.weight_bit) if num_classes > 0 else nn.Identity()
        self.act_out = QuantAct(nbits=cfg.activation_bit)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # LSQ: model_quant kept for API compatibility (does nothing)
    def model_quant(self):
        pass  # LSQ modules are always quantized with learnable alpha

    # freeze/unfreeze removed (LSQ keeps alpha learnable by default)

    def forward_features(self, x, current_abs_index=None, next_relative_index=None):
        if self.input_quant:
            x = self.qact_input(x)
        x = self.patch_embed(x)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.qact_embed(x)
        x = x + self.qact_pos(self.pos_embed)
        x = self.qact1(x)
        if current_abs_index is not None and next_relative_index is not None:
            if current_abs_index.size(1) != next_relative_index.size(1):
                current_abs_index = torch.gather(current_abs_index, 1, next_relative_index)
            x = torch.gather(x, 1, current_abs_index.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 0]
        x = self.qact2(x)
        return self.pre_logits(x)

    def forward(self, x, current_abs_index=None, next_relative_index=None):
        x = self.forward_features(x, current_abs_index, next_relative_index)
        x = self.head(x)
        return self.act_out(x)

# factory helpers

# -------------------------------------------------------------
def _build_deit_LSQ(embed_dim, depth, num_heads, cfg, **kw):
    return VisionTransformer(
        patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=4, qkv_bias=True, norm_layer=None,
        input_quant=True, cfg=cfg, **kw)

# Tiny -----------------------------------------------------------------
def deit_tiny_patch16_224_lsq(pretrained=False, cfg=None, **kw):
    model = _build_deit_LSQ(192, 12, 3, cfg, **kw)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True)
        model.load_state_dict(ckpt["model"], strict=False)   # float → LSQ
    return model

# Small ----------------------------------------------------------------
def deit_small_patch16_224_lsq(pretrained=False, cfg=None, **kw):
    model = _build_deit_LSQ(384, 12, 6, cfg, **kw)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True)
        model.load_state_dict(ckpt["model"], strict=False)
    return model

# Base -----------------------------------------------------------------
def deit_base_patch16_224_lsq(pretrained=False, cfg=None, **kw):
    model = _build_deit_LSQ(768, 12, 12, cfg, **kw)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)
        model.load_state_dict(ckpt["model"], strict=False)
    return model

@torch.no_grad()
def lsq_initialize(model: torch.nn.Module, device="cuda"):

    for m in model.modules():
        if hasattr(m, "initialize_scale") and callable(m.initialize_scale):
            m.initialize_scale(device)

    model.eval()

    dummy = torch.randn(4, 3, 224, 224, device=device)
    _ = model(dummy)

    print("LSQ scale initialization done.")

