import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from .cswin import CSWinBlock
from .cvt import Block as CvTBlock
from .layers import *
from einops.layers.torch import Rearrange
from einops import rearrange


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
from torchvision import models
class FCN(nn.Module):
    def __init__(self, in_channels=3, downsample=False, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3:
          newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
        #                           nn.BatchNorm2d(128), nn.ReLU())
        # 这里把layer3,layer4的stride 设置为1,分辨率也就不变了
        if not downsample:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n or 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv1' in n or 'downsample.0' in n:
                    m.stride = (1, 1)
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class BaseEncoder(nn.Module):
    def __init__(self, in_channels=3, downsample=False):
        super(BaseEncoder, self).__init__()        
        self.FCN = FCN(in_channels,downsample=downsample, pretrained=True)
        # initialize_weights( self.SiamSR, self.resCD, self.CotSR, self.classifierCD, self.classifier1, self.classifier2)
    
    
    
    def base_forward(self, x):
       
        x = self.FCN.layer0(x) #size:1/4
        x = self.FCN.maxpool(x) #size:1/4
        x = self.FCN.layer1(x) #size:1/4
        x = self.FCN.layer2(x) #size:1/8
        x = self.FCN.layer3(x) #size:1/16
        x = self.FCN.layer4(x) #size:1/32
        return x

    def forward(self, x1, x2):
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        return x1, x2


    def __init__(self, in_channels=3,downsample=False, IA_cfg=[0, 0, 1, 2, 2]):
        super(IAEncoder, self).__init__()        
        self.FCN = FCN(in_channels,downsample=downsample, pretrained=True)
        self.ia = self._make_ia_layer(IA_cfg)
        
    def _make_ia_layer(self, IA_cfg):
        layers = []
        for i in IA_cfg:
            layers.append(IAList[i]())
        return layers
    
    def base_forward(self, x1, x2):
        x1 = self.FCN.layer0(x1) #size:1/4
        x2 = self.FCN.layer0(x2) #size:1/4
        x1 = self.FCN.maxpool(x1) #size:1/4
        x2 = self.FCN.maxpool(x2) #size:1/4
        x1, x2 = self.ia[0](x1,x2)
        x1 = self.FCN.layer1(x1) #size:1/4
        x2 = self.FCN.layer1(x2) #size:1/4
        x1, x2 = self.ia[1](x1,x2)
        x1 = self.FCN.layer2(x1) #size:1/8
        x2 = self.FCN.layer2(x2) #size:1/8
        x1, x2 = self.ia[2](x1,x2)
        x1 = self.FCN.layer3(x1) #size:1/16
        x2 = self.FCN.layer3(x2) #size:1/16
        x1, x2 = self.ia[3](x1,x2)
        x1 = self.FCN.layer4(x1)
        x2 = self.FCN.layer4(x2)
        x1, x2 = self.ia[4](x1,x2)
        return x1, x2
    
    def forward(self, x1, x2):
        x1, x2 = self.base_forward(x1, x2)
        return x1, x2, [x1, x2]
CHANNELS=[64,128,256, 512]   

class BaseHead(nn.Module):
    def __init__(self, in_channel,num_classes=7) -> None:
        super().__init__()
        self.classifier1 = nn.Conv2d(in_channel, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(in_channel, num_classes, kernel_size=1)
        
        self.classifierCD = nn.Sequential(nn.Conv2d(in_channel, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
    
    def forward(self, change, x1, x2):
        change = self.classifierCD(change)
        x1 = self.classifier1(x1)
        x2 = self.classifier2(x2)       
        return change, x1, x2
    
class Neck(nn.Module):
    def __init__(self,embed_dim, mid_dim=128) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(embed_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(mid_dim), nn.ReLU())
        self.resCD = self._make_layer(ResBlock, mid_dim*2, mid_dim, 6, stride=1)# 1227: CAEncoder 输出的change 只有一个元素，所以用CAEncoder时不*2
        
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )
        
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    def base_forward(self, x1, x2):
        x1 = self.head(x1)
        x2 = self.head(x2)
        change = [x1, x2]
        change = torch.cat(change, 1)
        change = self.resCD(change)
        return change, x1, x2

    def forward(self, x1, x2, change):
        return self.base_forward(self, x1, x2, change)

class CATNeck(nn.Module):
    def __init__(self,embed_dim, mid_dim=128) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(embed_dim, mid_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(mid_dim), nn.ReLU())
        self.resCD = self._make_layer(ResBlock, mid_dim*2, mid_dim, 6, stride=1)# 1227: CAEncoder 输出的change 只有一个元素，所以用CAEncoder时不*2
        
        
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )
        
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    def forward(self, x1, x2, change):
        change = [self.head(x) for x in change]
        change = torch.cat(change, 1)
        change = self.resCD(change)
        x1 = self.head(x1)
        x2 = self.head(x2)
        return change, x1, x2

class CvTNeck(Neck):
    def __init__(self, embed_dim, mid_dim=128, heads=8, depth=2) -> None:
        super().__init__(embed_dim, mid_dim)
        self.conv_embed = nn.Sequential(
            nn.Conv2d(mid_dim*3, mid_dim*3, 3, 1, 1),
            nn.BatchNorm2d(mid_dim*3),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(mid_dim*3)
        ) 
        self.sa = nn.ModuleList(
            [CvTBlock(
                dim_in=mid_dim*3,  dim_out=mid_dim*3, num_heads=heads, with_cls_token=False)#, method='va')
            for i in range(depth)])
        
    def forward(self, x1, x2, change):
        x1, x2, change = self.base_forward(x1, x2, change)
        b, c, h, w = x1.shape
        token = torch.cat([x1, change, x2],1)
        token = self.conv_embed(token)
        for blk in self.sa:
            token = blk(token, h, w)
        x = rearrange(token, 'b (h w) c -> b c h w', h=h)
        x1, change, x2 = torch.chunk(x, dim=1,chunks=3)
        return change, x1, x2

class CSANeck(Neck):
    def __init__(self, embed_dim, mid_dim=128, bs=8 , heads=8, img_size=512, depth=2) -> None:
        super().__init__(embed_dim=embed_dim)
        self.conv_embed = nn.Sequential(
            nn.Conv2d(mid_dim*3, mid_dim*3, 3, 1, 1),
            nn.BatchNorm2d(mid_dim*3),
            Rearrange('b c h w -> b (h w) c', h = img_size//bs, w = img_size//bs),
            nn.LayerNorm(mid_dim*3)
        ) 
        self.sa = nn.ModuleList(
            [CSWinBlock(
                dim=mid_dim*3, num_heads=heads, reso=img_size//bs, split_size=2)
            for i in range(depth)])
        

    def forward(self, x1, x2):
        x1, x2, change = self.base_forward(x1, x2)
        b, c, h, w = x1.shape
        # change = [self.head(x) for x in change]
        # change = torch.cat(change, 1)
        # change = self.resCD(change)
        # x1 = self.head(x1)
        # x2 = self.head(x2)
        token = torch.cat([x1, change, x2],1)
        token = self.conv_embed(token)
        for blk in self.sa:
            token = blk(token)
        x = rearrange(token, 'b (h w) c -> b c h w', h=h)
        x1, change, x2 = torch.chunk(x, dim=1,chunks=3)
        return change, x1, x2
        
class CSABlock(nn.Module):
    def __init__(self, embed_dim, bs=8 , heads=8, img_size=512, depth=2) -> None:
        super().__init__()
        self.conv_embed = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim*2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim*2),
            Rearrange('b c h w -> b (h w) c', h = img_size//bs, w = img_size//bs),
            nn.LayerNorm(embed_dim*2)
        ) 
        self.sa = nn.ModuleList(
            [CSWinBlock(
                dim=embed_dim*2, num_heads=heads, reso=img_size//bs, split_size=2)
            for i in range(depth)])
    
    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        token = torch.cat([x1, x2],1)
        token = self.conv_embed(token)
        for blk in self.sa:
            token = blk(token)
        x = rearrange(token, 'b (h w) c -> b c h w', h=h)
        x1, x2 = torch.chunk(x, dim=1,chunks=2)
        return x1, x2
   
class BaseSCD(nn.Module):
    def __init__(self,in_channels=3, num_classes=7) -> None:
        super().__init__()    
        self.encoder = BaseEncoder(in_channels, downsample=False)
        self.neck = Neck(embed_dim=512, mid_dim=128)
        self.head = BaseHead(128, num_classes)
        
    def forward(self, x1, x2):
        x_size = x1.size()        
        x1, x2, change = self.encoder(x1,x2)
        x1, x2, change = self.neck(x1, x2, change)
        change, x1, x2 = self.head(x1,x2, change)
        return F.interpolate(change, x_size[2:], mode='bilinear', align_corners=True), F.interpolate(x1, x_size[2:], mode='bilinear', align_corners=True), F.interpolate(x2, x_size[2:], mode='bilinear', align_corners=True)

class SCDNetest(nn.Module):
    def __init__(self,in_channels=3, num_classes=7,IA_cfg=[0, 0, 1, 2, 2]) -> None:
        super().__init__()    
        self.encoder = BaseEncoder(in_channels, downsample=False)
        self.neck = CATNeck(embed_dim=512, mid_dim=128)
        self.head = BaseHead(128, num_classes)
        
    def forward(self, x1, x2):
        x_size = x1.size()        
        x1, x2, change = self.encoder(x1,x2)
        x1, x2, change = self.neck(x1, x2, change)
        change, x1, x2 = self.head(x1,x2, change)
        return F.interpolate(change, x_size[2:], mode='bilinear', align_corners=True), F.interpolate(x1, x_size[2:], mode='bilinear', align_corners=True), F.interpolate(x2, x_size[2:], mode='bilinear', align_corners=True)

class SCDNet(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()    
        en_name, en_arg = cfg['MODEL']['ENCODER']
        ne_name, ne_arg = cfg['MODEL']['NECK']
        de_name, de_arg = cfg['MODEL']['HEAD']
        self.encoder = eval(en_name)(*en_arg)
        self.neck = eval(ne_name)(*ne_arg)
        self.head = eval(de_name)(*de_arg)
        
    def forward(self, x1, x2):
        x_size = x1.size()        
        x1, x2 = self.encoder(x1,x2)
        x1, x2, change = self.neck(x1, x2)
        change, x1, x2 = self.head(x1,x2, change)
        return F.interpolate(change, x_size[2:], mode='bilinear', align_corners=True), F.interpolate(x1, x_size[2:], mode='bilinear', align_corners=True), F.interpolate(x2, x_size[2:], mode='bilinear', align_corners=True)

        
        
if __name__ == '__main__':
    device = 'cuda:1'
    # device = 'cpu'
    x1 = torch.rand(4,3,512,512).to(device)
    x2 = torch.rand(4,3,512,512).to(device)
    model = SCDNetest().to(device)
    change, x1, x2 = model(x1,x2)
    print(change.shape, x1.shape, x2.shape)