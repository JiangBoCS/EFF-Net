import numpy as np
from numpy.random import RandomState
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import Module
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import sys
import torch.nn.init as init
from typing import List, Tuple
# from quaternion_layers import QuaternionTransposeConv,QuaternionConv, QuaternionLinearAutograd

class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(Linear, self).__init__()

        self.Linear = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self, x):
        out = self.Linear(x)
        return out

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        drop_rate = 0.
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return self.pos_drop(x)

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H, W):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

class DHA(nn.Module):#LCSA_Layer
    def __init__(self, channel_num, reduction=16):
        super(DHA, self).__init__()

        self.L0 = Linear(channel_num,channel_num)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Hash = nn.Sequential(
                nn.Conv2d(channel_num, channel_num // reduction, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Conv2d(channel_num // reduction, channel_num, 1, padding=0, bias=True),
                nn.Tanh()
        )
        
    def forward(self, x):

        y = self.avg_pool(x)
        threshold = self.L0(y.squeeze(-1).transpose(2,1)).transpose(2,1).unsqueeze(-1)
        
        w = torch.abs(self.Hash(y))
        
        zero = torch.zeros_like(w)
        one = torch.ones_like(w)
        y = torch.where(w > threshold, one, zero)

        return x * y
        
class DF(nn.Module):
    '''Frequency-Hierarchy module'''

    def __init__(self, channel_num):
        super(DF, self).__init__()

        self.C0 = nn.Sequential(
             nn.Conv2d(channel_num, channel_num//3, groups=channel_num//3, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(inplace=True))
             
        self.C1 = nn.Sequential(
             nn.Conv2d(channel_num, channel_num//3, groups=channel_num//3, kernel_size=3, stride=1, padding=2, dilation = 2),
             nn.LeakyReLU(inplace=True))    

        self.C2 = nn.Sequential(
             nn.Conv2d(channel_num, channel_num//3, groups=channel_num//3, kernel_size=3, stride=1, padding=3, dilation = 3),
             nn.LeakyReLU(inplace=True))

        self.R = nn.GELU()

    def forward(self, x):
        l = self.R(self.C2(x))
        m = self.R(self.C1(x) - l)
        h = self.R(self.C0(x) - self.C1(x))
        return l, m, h   

class EFF(nn.Module):
    '''Frequency enhancement module'''

    def __init__(self, dim=32, out_dim=128):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, dim),
                                     nn.GELU())
        self.DF = DF(dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(dim, out_dim))

    def forward(self, x, H, W):
        # bs x hw x c
#        short = x
        bs, hw, c = x.size()
        x = self.linear1(x)
        
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        sht = x
        # bs,hidden_dim,32x32
        l, m, h = self.DF(x)
        x = torch.cat((l, m, h), dim = 1)
        x = self.dwconv(x)
        x = x + sht

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)

        x = self.linear2(x)

        return x
        
class AFEBlock(nn.Module):
    def __init__(self, dim=3, out_dim = 6):
        super().__init__()

        self.LN1 = nn.LayerNorm(dim)
        self.attn = DHA(dim)

        self.LN2 = nn.LayerNorm(dim)
        self.EFF = EFF(dim, out_dim)

    def forward(self, x, H, W):
        B, L, C = x.shape

        shortcut = x
        x = self.LN1(x)
        xx = self.attn(x.contiguous().view(B, C, H, W))
        x = xx.contiguous().view(B, H * W, C) + shortcut 
        sht = x

        # EFF
        x = self.EFF(self.LN2(x)+ sht, H, W)
        return x

class EFF_Net(nn.Module):
    def __init__(self, in_channel=3, dim=3):
        super().__init__()

        self.In = InputProj(in_channel, dim)

        self.FETBlock1 = AFEBlock(dim, dim)
        self.D1 = Downsample(dim, dim)

        self.FETBlock2 = AFEBlock(dim, dim*2)
        self.D2 = Downsample(dim*2, dim*2)

        self.FETBlock3 = AFEBlock(dim*2, dim * 4)
        self.D3 = Downsample(dim * 4, dim * 4)

        self.FETBlock4 = AFEBlock(dim* 4, dim * 8)
        self.D4 = Downsample(dim * 8, dim * 8)

        self.BNeck = AFEBlock(dim* 8, dim * 16)

        self.U6 = Upsample(dim*16, dim * 8)
        self.FETBlock6 = AFEBlock(dim * 16, dim * 8)

        self.U7 = Upsample(dim * 8, dim*4)
        self.FETBlock7 = AFEBlock(dim * 8, dim*4)

        self.U8 = Upsample(dim * 4, dim*2)
        self.FETBlock8 = AFEBlock(dim * 4, dim*2)

        self.U9 = Upsample(dim * 2, dim)
        self.FETBlock9 = AFEBlock(dim * 2, dim)

        self.Out = OutputProj(dim,3)


    def forward(self, x):
        H, W = x.shape[2:]
        short_x = x
        x = self.In(x)

        conv1 = self.FETBlock1(x, H, W)
        pool1 = self.D1(conv1, H, W)

        conv2 = self.FETBlock2(pool1, H//2, W//2)
        pool2 = self.D2(conv2, H//2, W//2)

        conv3 = self.FETBlock3(pool2, H//4, W//4)
        pool3 = self.D3(conv3, H//4, W//4)

        conv4 = self.FETBlock4(pool3, H//8, W//8)
        pool4 = self.D4(conv4, H//8, W//8)

        conv5 = self.BNeck(pool4, H//16, W//16)

        up6 = self.U6(conv5, H//16, W//16)
        up6 = torch.cat([up6, conv4], 2)
        conv6 = self.FETBlock6(up6, H//8, W//8)

        up7 = self.U7(conv6, H//8, W//8)
        up7 = torch.cat([up7, conv3], 2)
        conv7 = self.FETBlock7(up7, H//4, W//4)

        up8 = self.U8(conv7, H//4, W//4)
        up8 = torch.cat([up8, conv2], 2)
        conv8 = self.FETBlock8(up8, H//2, W//2)

        up9 = self.U9(conv8, H//2, W//2)
        up9 = torch.cat([up9, conv1], 2)
        conv9 = self.FETBlock9(up9, H, W)
        x = self.Out(conv9, H, W) + short_x
        return x

if __name__ == '__main__':

    from ptflops import get_model_complexity_info
    import time
    
    x = torch.randn(1,3,128,128)
    M = EFF_Net(dim=18)
    y = M(x)
    print(y.shape)

    flops, params = get_model_complexity_info(M, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    
#    dim=54; Params: 9.70 M; Flops:  15.68 GMac
#    dim=36; Params: 4.34 M; Flops:  7.18 GMac
#    dim=18; Params: 1.10 M; Flops:  1.96 GMac


    
    
