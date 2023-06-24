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
    def __init__(self, input_channels, output_channels, stride=1, bias=False, output=False):
        super().__init__()
        act = nn.Tanh() if output else nn.ReLU()
        self.conv = nn.Sequential(
                    LayerNorm2d(input_channels),
                    act,
                    nn.Conv2d(input_channels,output_channels,kernel_size=3, stride=stride, padding=1, bias=bias),
                    LayerNorm2d(output_channels),
                    act,
                    nn.Conv2d(output_channels,output_channels,kernel_size=1, stride=stride, padding=0, bias=bias)
                )
        # skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                    LayerNorm2d(input_channels),
                    act,
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
    def __init__(self, dim, n_heads, reduction_ratio, mlp_expansion, depth_prob=0., img_size=28):
        super().__init__()
        # multihead attention residual addition and mixMLP
        self.transformer = nn.Sequential(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(dim),
                    EfficientMultiHeadAttention(dim, n_heads, reduction_ratio, img_size),
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
    def __init__(self, low_res_ch, high_res_ch, n_heads, reduction_ratio, mlp_expansion, depth_prob=0., img_size=28):
        super().__init__()
        self.embedding = nn.Conv2d(low_res_ch, high_res_ch, kernel_size=1)

        # attention decoder
        self.bn_l = LayerNorm2d(low_res_ch)
        self.bn_h = LayerNorm2d(high_res_ch)
        self.attn = EfficientMultiHeadAttention(high_res_ch, n_heads, reduction_ratio, img_size)

        # residual addition and mixMLP
        self.mlp = ResidualAdd(nn.Sequential(
                    LayerNorm2d(high_res_ch),
                    MixMLP(high_res_ch, expansion=mlp_expansion),
                    StochasticDepth(p=depth_prob, mode="batch")
                )
            )

    def forward(self, x1, x2):
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
        #x1: low-res, x2: high-res
        x1 = self.bn_l(x1)
        x2 = self.bn_h(x2)

        # embed dimensions
        x1 = self.embedding(x1)

        # calculate residue
        x1_shape = x1.shape
        res = x1.expand(1,x1_shape[0],x1_shape[1],x1_shape[2])
        res = F.interpolate(res, size=x2.shape[-2:], mode='bilinear', align_corners=True)[0]

        # attention decoder
        out = self.attn(x1, x2)
        
        # add residue
        out = out + res

        # multilayer perceptron layer
        out = self.mlp(out)
        return out
    

#######################################
# Auxiliar Networks layers class
#######################################
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "c h w -> h w c")
        x = super().forward(x)
        x = rearrange(x, "h w c -> c h w")
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
    def __init__(self, embed_dim: int, num_heads: int = 8, reduction_ratio: int = 1, reduced_size: int = 28):
        super().__init__()
        # positional information
        self.pos_embedding = nn.Parameter(torch.randn(embed_dim, reduced_size, reduced_size)*0.02)
        # reducer convolution
        self.reducer = nn.Sequential(
            nn.Conv2d(
                embed_dim, embed_dim, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(embed_dim),
        )
        # multihead attention module
        self.att = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=False
        )

    def forward(self, x, q=None):
        _, h, w = x.shape if q is None else q.shape
        # add position embedding
        x += self.pos_embedding
        # reduce dimensions
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