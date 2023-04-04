#!/usr/bin/env python
# coding: utf-8
# Daniel Bandala @ nov-2022
# Transformer encoder-decoder model
import torch
from torch import nn
from transformer_module import TransformerEncoder, TransformerDecoder, ResidualBasicBlock

#######################################
# Diffusion tensor model
#######################################
class DiffusionTensorModel(nn.Module):
    """ UTNet architecture based on Vision transformers
    ----------
    img_size : int
        Both height and the width of the image (it is a square).
    patch_size : int
        Both height and the width of the patch (it is a square).
    in_chans : int
        Number of input channels.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.
    pos_drop : nn.Dropout
        Dropout layer.
    blocks : nn.ModuleList
        List of `Block` modules.
    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            in_chans=6,
            out_chans=1,
            img_size=140,
            embed_dim=64,
            n_heads=[1,2,4,8],
            mlp_ratio=[2,4,8,16],
            reduction_ratio=1,
            depth_prob=0.,
            tanh_output=False,
    ):
        super().__init__()
        # layer channels
        self.out_chans = out_chans
        mid_channels = embed_dim//2
        layer1_channels = embed_dim*2
        layer2_channels = embed_dim*4
        layer3_channels = embed_dim*8
        layer4_channels = embed_dim*16

        # spatial dimension by layer
        size_x2 = img_size//2
        size_x4 = img_size//4
        size_x8 = img_size//8
        size_x16 = img_size//16

        # input layer
        self.inconv = nn.Sequential(
            ResidualBasicBlock(in_chans,embed_dim),
            ResidualBasicBlock(embed_dim,embed_dim)
        )
        
        # Encoder stage
        self.transformer_encoder_1 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBasicBlock(embed_dim,layer1_channels),
            TransformerEncoder(layer1_channels, n_heads[0], reduction_ratio, mlp_ratio[0], depth_prob, size_x2)
        )
        self.transformer_encoder_2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBasicBlock(layer1_channels,layer2_channels),
            TransformerEncoder(layer2_channels, n_heads[1], reduction_ratio, mlp_ratio[1], depth_prob, size_x4)
        )
        self.transformer_encoder_3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBasicBlock(layer2_channels,layer3_channels),
            TransformerEncoder(layer3_channels, n_heads[2], reduction_ratio, mlp_ratio[2], depth_prob, size_x8)
        )
        self.transformer_encoder_4 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBasicBlock(layer3_channels,layer4_channels),
            TransformerEncoder(layer4_channels, n_heads[3], reduction_ratio, mlp_ratio[3], depth_prob, size_x16)
        )

        # Decoder stage
        self.transformer_decoder_1 = TransformerDecoder(layer4_channels, layer3_channels, n_heads[3], reduction_ratio, mlp_ratio[3], depth_prob, size_x16)
        self.res1 = ResidualBasicBlock(layer4_channels,layer3_channels)
        self.transformer_decoder_2 = TransformerDecoder(layer3_channels, layer2_channels, n_heads[2], reduction_ratio, mlp_ratio[2], depth_prob, size_x8)
        self.res2 = ResidualBasicBlock(layer3_channels,layer2_channels)
        self.transformer_decoder_3 = TransformerDecoder(layer2_channels, layer1_channels, n_heads[1], reduction_ratio, mlp_ratio[1], depth_prob, size_x4)
        self.res3 = ResidualBasicBlock(layer2_channels,layer1_channels)

        # upsample with inverse convolution
        self.upsample = nn.ConvTranspose2d(layer1_channels, embed_dim, 2,2)
        
        # output layers
        self.outconv = nn.Sequential(
            ResidualBasicBlock(2*embed_dim, embed_dim),
            ResidualBasicBlock(embed_dim, mid_channels),
            ResidualBasicBlock(mid_channels, out_chans, output=tanh_output)
        )


    def forward(self, x):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(samples, in_chans, img_size, img_size)`.
        Returns
        -------
        Feature maps : torch.Tensor
            Feature maps images - `(img_size, img_size)`.
        """
        # input layer execution
        xi = self.inconv(x)

        # encoder layers
        x1 = self.transformer_encoder_1(xi)
        x2 = self.transformer_encoder_2(x1)
        x3 = self.transformer_encoder_3(x2)
        x4 = self.transformer_encoder_4(x3)

        # decoder layers
        x = self.transformer_decoder_1(x4, x3)
        x = self.res1(torch.cat((x,x3),0))
        x = self.transformer_decoder_2(x, x2)
        x = self.res2(torch.cat((x,x2),0))
        x = self.transformer_decoder_3(x, x1)
        x = self.res3(torch.cat((x,x1),0))
        
        # upsample layer
        x = self.upsample(x)

        # concatenate output with high resolution features
        x = torch.cat((x,xi),0)
        # output layer
        x = self.outconv(x)

        # return image tensor
        return x if self.out_chans>1 else x[0]