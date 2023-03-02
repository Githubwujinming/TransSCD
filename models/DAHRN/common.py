
import torch 
import torch.nn as nn
from .resnet import build_backbone
from .hrnet import hrnet18, hrnet32, hrnet48
from .fcs import FCSiamConc
from .VAN import van18, Attention, van_tiny

from .crossat import cat18, cat32


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SCAM(nn.Module):
    def __init__(self,dim, out_c) -> None:
        #[256,160,]
        super().__init__()
        self.ca = ChannelAttention(dim, ratio=16)
        # self.conv0 = nn.Conv2d(dim,dim,1,1)
        # self.ca1 = ChannelAttention(channel_list[0], ratio=16 // 4)
        self.sp = Attention(dim)
        self.conv = nn.Conv2d(dim,out_c,1,1)

    def forward(self,x):
        # x = x + self.conv0(self.ca(x)*x)
        x = self.ca(x)*x
        x = self.sp(x)
        x = self.conv(x)
        return x

class convout(nn.Module):
    def __init__(self, in_channels,nc=2,bn_mom = 0.0003):
        super().__init__()
        self.conv_out = nn.Sequential(
                        nn.Conv2d(in_channels,in_channels//2,kernel_size=3, stride=1, padding=1),
                        torch.nn.BatchNorm2d(in_channels//2, momentum=bn_mom),
                        nn.GELU(),
                        nn.Conv2d(in_channels//2,nc,kernel_size=3,stride=1,padding=1),
                        # nn.Softmax(dim=1),
                        )
    def forward(self, input):
        return self.conv_out(input)

         


class SCAT(nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, bilinear=False, upsample = False,bn_mom = 0.0003):
        super().__init__() ##parent's init func
        #[256, 160, 64, 32]
        self.do_upsample = upsample
        self.upsample = UP(in_ch=in_chn_high, bilinear=bilinear)
        self.sca = SCAM(in_chn_high+in_chn_low, out_chn)
    
    def forward(self,x,y):
        if self.do_upsample:
            x = self.upsample(x)
        x = torch.cat((x,y),1)#x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.sca(x)


class CAT(nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, bilinear=False, upsample = False,bn_mom = 0.0003):
        super().__init__() ##parent's init func
        #[256, 160, 64, 32]
        self.do_upsample = upsample
        self.upsample = UP(in_ch=in_chn_high, bilinear=bilinear)
        self.conv2d=nn.Sequential(
            nn.Conv2d(in_chn_high + in_chn_low, out_chn, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            nn.GELU(),
        )
    
    def forward(self,x,y):
        if self.do_upsample:
            x = self.upsample(x)
        x = torch.cat((x,y),1)#x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.conv2d(x)

class TCAT(nn.Module):
    def __init__(self, in_chn_high, bn_mom = 0.0003):
        super().__init__() ##parent's init func
        #[256, 160, 64, 32]
        self.conv2d=nn.Sequential(
            nn.Conv2d(in_chn_high*3, in_chn_high, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_chn_high, momentum=bn_mom),
            nn.GELU(),
        )
    
    def forward(self,y1, y2, c):
        x = torch.cat((y1, y2, c),1)#x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.conv2d(x)

class TSCAT(nn.Module):
    def __init__(self, in_chn_high):
        super().__init__() ##parent's init func
        #[256, 160, 64, 32]
        self.sca = SCAM(in_chn_high*3, in_chn_high)

    
    def forward(self,y1, y2, c):
        x = torch.cat((y1, y2, c),1)#x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.sca(x)



class UP(nn.Module):
    def __init__(self, in_ch,out_ch=None, bilinear=False):
        super().__init__()
        if out_ch == None:
            out_ch = in_ch
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,1,stride=1,padding=0),
                nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True),)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x



CHANNEL_LIST = {
    'resnet18': [144,72,36,18],
    'resnet32': [256, 128, 64, 32],
    'van18':[144,72,36,18],
    'vtiny':[256, 160, 64, 32],
    'cat18':[144,72,36,18],
    'cat32':[256, 128, 64, 32],
    'hrnet18':[144, 72, 36, 18],
    'hrnet32':[256, 128, 64, 32],
    'hrnet48':[364, 192, 96, 48],
    'fcsc':[128,64,32,16],
}

BACKBONE = {
    'resnet18':build_backbone,
    'resnet34':build_backbone,
    'resnet32':build_backbone,
    'resnet50':build_backbone,
    'van18':van18,
    'vtiny':van_tiny,
    'hrnet18':hrnet18,
    'hrnet32':hrnet32,
    'hrnet48':hrnet48,
    'fcsc':FCSiamConc,
    'cat18':cat18,
    'cat32':cat32,
}



def get_encoder(arch,output_stride, BatchNorm=nn.BatchNorm2d, in_c=3):
    encoder=None
    if arch not in CHANNEL_LIST.keys():
        NotImplementedError('no such encoder')
    channel_list = CHANNEL_LIST[arch]
    channel_list.reverse()
    if 'resnet' in arch:
        encoder = BACKBONE[arch](arch,output_stride, BatchNorm,in_c, channel_list=channel_list)
    elif 'fcs' in arch:
        encoder = BACKBONE[arch](in_c)
    else:
        encoder = BACKBONE[arch]()
    return encoder, channel_list

        