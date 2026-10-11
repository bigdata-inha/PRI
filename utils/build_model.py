#transform timm vits to versions that can stop feeding forward specific patches

from types import MethodType
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Attention,Block

# def forward(self, x):
#     B, N, C = x.shape
#     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

#     attn = (q @ k.transpose(-2, -1)) * self.scale
#     attn = attn.softmax(dim=-1)
#     attn = self.attn_drop(attn)

#     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#     x = self.proj(x)
#     x = self.proj_drop(x)
#     return x

def vit_attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x,attn

# class Block(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads,
#             mlp_ratio=4.,
#             qkv_bias=False,
#             drop=0.,
#             attn_drop=0.,
#             init_values=None,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
#         self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x

def vit_block_forward(self,x):
    x_out,attn_out=self.attn(self.norm1(x))
    x = x + (x_out)
    x = x + (self.mlp(self.norm2(x)))
    return x,attn_out

# def forward_features(self, x):
#     x = self.patch_embed(x)
#     x = self._pos_embed(x)
#     x = self.norm_pre(x)
#     if self.grad_checkpointing and not torch.jit.is_scripting():
#         x = checkpoint_seq(self.blocks, x)
#     else:
#         x = self.blocks(x)
#     x = self.norm(x)
#     return x

# def forward_head(self, x, pre_logits: bool = False):
#     if self.global_pool:
#         x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
#     x = self.fc_norm(x)
#     return x if pre_logits else self.head(x)

# def forward(self, x):
#     x = self.forward_features(x)
#     x = self.forward_head(x)
#     return x

def vit_forward_features(self, x, current_abs_index, next_relative_index):
    B = x.shape[0]
    x = self.patch_embed(x)
    cls_tokens = self.cls_token.expand(
        B, -1, -1
    )  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    # sparse
    if next_relative_index.shape[1] == current_abs_index.shape[1]:
        pass
    else:
        current_abs_index=torch.gather(current_abs_index,1,next_relative_index)
        assert current_abs_index[0][0] == 0

    x = torch.gather(x, 1, current_abs_index.unsqueeze(-1).expand(-1, -1, x.size(-1)))

    x = self.pos_drop(x)
    attn_weights = []
    for blk in self.blocks:
        x, attn = blk(x)
        attn_weights.append(attn)
    attn_weights = torch.stack(attn_weights, dim=0)
    x = self.norm(x)[:, 0]

    return x,attn_weights,current_abs_index

def vit_forward(self, x, current_abs_index, next_relative_index):
    x, attn_out,current_abs_index = self.forward_features(x, current_abs_index, next_relative_index)
    x = self.head(x)
    return x,attn_out,current_abs_index





class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def build_model(name, Pretrained=True):
    """
    Get a vision transformer model.

    This will insert
    current_abs_index (the absolute index of current patches)
    and next_relative_index  (the relative index of patches to retain)
    to the original input of attention.forward, block.forward/forward_feature, and net.forward

    Currently support almost all quantization in timm.quantization.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    """
    net = timm.create_model(name,pretrained=Pretrained)

    for name, module in net.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(vit_attention_forward, module)
        if isinstance(module,Block):
            module.forward = MethodType(vit_block_forward, module)
            net.forward_features = MethodType(vit_forward_features, net)
            net.forward = MethodType(vit_forward, net)

    net = net.cuda()
    net.eval()
    return net
