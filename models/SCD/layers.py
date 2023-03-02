import torch.nn as nn
from torch.nn import functional as F
import torch
from typing import List, Optional
from torch import Tensor, reshape, stack
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
)


class PixelwiseLinear(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)



# 双时相图片的同一语义通道层合并，两个通道一组进行组卷积，得到语义特征大小的变化特征
class MixingBlock(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)

class MixingBlockT(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x:List) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack(x, dim=2)
        mixed = reshape(mixed, (x[0].shape[0], -1, x[0].shape[2], x[0].shape[3]))

        # Mixing:
        return self._convmix(mixed)

class MixingMaskAttentionBlock(Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: List[int],
        fout: List[int],
        generate_masked: bool = False,
    ):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


class Patch_Attention(nn.Module):
    def __init__(self, in_channels, reduction=8, pool_window=10, add_input=False):
        super(Patch_Attention, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        pool_h = h//self.pool_window
        pool_w = w//self.pool_window
        
        A = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        A = self.SA(A)
        
        A = F.interpolate(A, (h,w), mode='bilinear',align_corners=True)        
        output = x*A
        if self.add_input:
            output += x
        
        return output
    
class CDPA(nn.Module):
    def __init__(self, in_channels, reduction=8, pool_window=10, add_input=False):
        super(CDPA, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        b, c, h, w = x1.size()
        pool_h = h//self.pool_window
        pool_w = w//self.pool_window
        
        A1 = F.adaptive_avg_pool2d(x1, (pool_h, pool_w))
        A1 = self.SA(A1)
        
        A1 = F.interpolate(A1, (h,w), mode='bilinear',align_corners=True)        
        output1 = x1*A1
        if self.add_input:
            output1 += x1
        
        A2 = F.adaptive_avg_pool2d(x2, (pool_h, pool_w))
        A2 = self.SA(A2)
        
        A2 = F.interpolate(A2, (h,w), mode='bilinear',align_corners=True)        
        output2 = x2*A2
        if self.add_input:
            output2 += x2
        return output1, output2

class SCDPA(nn.Module):
    def __init__(self, in_channels, reduction=8, pool_window=10, add_input=False):
        super(SCDPA, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2, x):
        b, c, h, w = x.size()
        pool_h = h//self.pool_window
        pool_w = w//self.pool_window
        
        A = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        A = self.SA(A)
        
        A = F.interpolate(A, (h,w), mode='bilinear',align_corners=True)        
        output1 = x1*A
        output2 = x2*A
        if self.add_input:
            output1 += x1
            output2 += x2
        
        return output1, output2
# Copyright (c) Open-CD. All rights reserved.


class ChannelExchange(nn.Module):
    """
    channel exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super(ChannelExchange, self).__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        
        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]
        
        return out_x1, out_x2

class SpatialExchange(nn.Module):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """
    def __init__(self, p=1/2):
        super(SpatialExchange, self).__init__()
        assert p >= 0 and p <= 1
        self.p = int(1/p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        
        return out_x1, out_x2


class Aggregation_distribution(nn.Module):
    # Aggregation_Distribution Layer (AD)
    def __init__(self, 
                 channels, 
                 num_paths=2, 
                 attn_channels=None, 
                 act_cfg=nn.ReLU,
                 norm_cfg=nn.BatchNorm2d):
        super(Aggregation_distribution, self).__init__()
        self.num_paths = num_paths # `2` is supported.
        attn_channels = attn_channels or channels // 16
        attn_channels = max(attn_channels, 8)
        
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = norm_cfg(attn_channels)
        self.act = act_cfg()
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        attn = x.sum(1).mean((2, 3), keepdim=True)
        attn = self.fc_reduce(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        attn = self.fc_select(attn)
        B, C, H, W = attn.shape
        attn1, attn2 = attn.reshape(B, self.num_paths, C // self.num_paths, H, W).transpose(0, 1)
        attn1 = torch.sigmoid(attn1)
        attn2 = torch.sigmoid(attn2)
        return x1 * attn1, x2 * attn2


class TwoIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TwoIdentity, self).__init__()

    def forward(self, x1, x2):
        return x1, x2
