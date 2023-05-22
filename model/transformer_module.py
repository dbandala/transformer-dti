#!/usr/bin/env python
# coding: utf-8
# Daniel Bandala @ nov-2022
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torchvision.ops import StochasticDepth
# Borrowed some code from UTNet: https://github.com/yhygao/UTNet/blob/main/model/conv_trans_utils.py

#######################################
# Residual block class
#######################################
class ResidualBasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, bias=False):
        super().__init__()
        self.conv = nn.Sequential(
                    LayerNorm2d(input_channels),
                    nn.ReLU(),
                    nn.Conv2d(input_channels,output_channels,kernel_size=3, stride=stride, padding=1, bias=bias),
                    LayerNorm2d(output_channels),
                    nn.ReLU(),
                    nn.Conv2d(output_channels,output_channels,kernel_size=1, stride=stride, padding=0, bias=bias)
                )
        # skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                    LayerNorm2d(input_channels),
                    nn.ReLU(), 
                    nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return out


#######################################
# Transformer encoder class
#######################################
class TransformerEncoder(nn.Module):
    """Transformer block.
    Parameters
    ----------
    dim : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, reduction_ratio, mlp_expansion, depth_prob=0.):
        super().__init__()
        # multihead attention residual addition and mixMLP
        self.transformer = nn.Sequential(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(dim),
                    EfficientMultiHeadAttention(dim, n_heads, reduction_ratio),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(dim),
                    MixMLP(dim, expansion=mlp_expansion),
                    StochasticDepth(p=depth_prob, mode="batch")
                )
            )
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_patches, channels, height, width)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_patches, channels, height, width)`.
        """
        out = self.transformer(x)
        return out

#######################################
# Transformer decoder class
#######################################
class TransformerDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, heads, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=False):
        super().__init__()

        self.bn_l = LayerNorm2dExt(in_ch)
        self.bn_h = LayerNorm2dExt(out_ch)

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.attn = LinearAttention(in_ch, out_ch, heads=heads, dim_head=out_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        self.bn2 = LayerNorm2dExt(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        # expand dimension
        x1_shape = x1.shape
        x2_shape = x2.shape
        x1 = x1.expand(1,x1_shape[0],x1_shape[1],x1_shape[2])
        x2 = x2.expand(1,x2_shape[0],x2_shape[1],x2_shape[2])

        # calculate residue
        residue = F.interpolate(self.conv_ch(x1), size=x2.shape[-2:], mode='bilinear', align_corners=True)
        #x1: low-res, x2: high-res
        x1 = self.bn_l(x1)
        x2 = self.bn_h(x2)

        # linear attention
        out, q_k_attn = self.attn(x2, x1)
        
        out = out + residue
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out[0]

#######################################
# Auxiliar Networks layers class
#######################################
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "c h w -> h w c")
        x = super().forward(x)
        x = rearrange(x, "h w c -> c h w")
        return x

class LayerNorm2dExt(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

#######################################
# Transformer components
#######################################
class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, reduction_ratio: int = 1):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(embed_dim),
        )
        self.att = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=False
        )

    def forward(self, x, q=None):
        _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (sequence_length, embed_dim)
        reduced_x = rearrange(reduced_x, "c h w -> (h w) c")
        x = rearrange(x, "c h w -> (h w) c")
        # encoder-decoder condition
        query = x if q is None else rearrange(q, "c h w -> (h w) c")
        # get attention map
        attn = self.att(query, reduced_x, reduced_x)
        # reshape it back to (batch, embed_dim, height, width)
        out = rearrange(attn[0], "(h w) c -> c h w", h=h, w=w)
        return out

class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )


class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
                torch.randn((2*h-1) * (2*w-1), num_heads)*0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij")) # 2, h, w
        coords_flatten = torch.flatten(coords, 1) # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1) # hw, hw
    
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h, self.w, self.h*self.w, -1) #h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H//self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W//self.w, dim=1) #HW, hw, nH
        # interpolate dimension if pixels are needed
        if (H%self.h!=0 or W%self.w!=0):
            relative_position_bias_expanded = relative_position_bias_expanded.permute(2,3,0,1)
            relative_position_bias_expanded = F.interpolate(relative_position_bias_expanded, size=[H,W], mode='bilinear', align_corners=True)
            relative_position_bias_expanded = relative_position_bias_expanded.permute(2,3,0,1)
        # reshape postional embedding vector
        relative_position_bias_expanded = relative_position_bias_expanded.view(H*W, self.h*self.w, self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)
        return relative_position_bias_expanded

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x): 
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out 

class LinearAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=False):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos
        
        # depthwise conv is slightly better than conv1x1
        #self.to_kv = nn.Conv2d(in_dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_q = nn.Conv2d(out_dim, self.inner_dim, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
       
        self.to_kv = depthwise_separable_conv(in_dim, self.inner_dim*2)
        self.to_q = depthwise_separable_conv(out_dim, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, out_dim)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            #self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, q, x):
        B, C, H, W = x.shape # low-res feature shape
        BH, CH, HH, WH = q.shape # high-res feature shape

        k, v = self.to_kv(x).chunk(2, dim=1) #B, inner_dim, H, W
        q = self.to_q(q) #BH, inner_dim, HH, WH

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=HH, w=WH)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        
        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(HH, WH) #HH, WH
            q_k_attn += relative_position_bias
            #rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, HH, WH, self.dim_head)
            #q_k_attn = q_k_attn + rel_attn_h + rel_attn_w
       
        q_k_attn *= self.scale
        q_k_attn = torch.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HH, w=WH, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn