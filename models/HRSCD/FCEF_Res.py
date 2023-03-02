'''
R. Caye Daudt, B. Le Saux, and A. Boulch, “Fully convolutional siamese networks for change detection,” in Proceedings - International Conference on Image Processing, ICIP, 2018, pp. 4063–4067.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes=None, stride=1, dilation=1, use_1x1=False, BatchNorm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        if planes==None:
            planes = inplanes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = BatchNorm(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.use_1x1 = use_1x1
        if use_1x1:
            self.conv3=nn.Conv2d(
                inplanes,planes,kernel_size=1,stride=stride) 

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_1x1:
            identity = self.conv3(x)
        out += identity
        out = self.relu(out)

        return out

class HRSCD2(nn.Module):
    def __init__(self, in_dim=3, nc=7):
        super(HRSCD2, self).__init__()

        self.conv_block_1 = nn.Sequential(
            BasicBlock(inplanes=in_dim*2, planes=16, dilation=1, use_1x1=True),
            BasicBlock(inplanes=16, planes=16, dilation=1),
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            BasicBlock(inplanes=16, planes=32, dilation=1, use_1x1=True),
            BasicBlock(inplanes=32, planes=32, dilation=1),
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            BasicBlock(inplanes=32, planes=64, dilation=1, use_1x1=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
            BasicBlock(inplanes=64, planes=64, dilation=1),
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            BasicBlock(inplanes=64, planes=128, dilation=1, use_1x1=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
            BasicBlock(inplanes=128, planes=128, dilation=1),
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
        )
        self.conv_block_5 = nn.Sequential(
            BasicBlock(inplanes=256, planes=128, dilation=1, use_1x1=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
            BasicBlock(inplanes=128, planes=64, dilation=1, use_1x1=True),
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
        )
        self.conv_block_6 = nn.Sequential(
            BasicBlock(inplanes=128, planes=64, dilation=1, use_1x1=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
            BasicBlock(inplanes=64, planes=32, dilation=1, use_1x1=True),
        )

        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=32, planes=32, dilation=1),
        )
        self.conv_block_7 = nn.Sequential(
            BasicBlock(inplanes=64, planes=32, dilation=1, use_1x1=True),
            BasicBlock(inplanes=32, planes=16, dilation=1, use_1x1=True),
        )

        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=16, planes=16, dilation=1),
        )
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=nc, kernel_size=1, padding=0),
        )
        self.apply(self._init_weights)

    def _init_weights(self,m):
       if isinstance(m, nn.Linear):
          nn.init.xavier_normal_(m.weight)
          nn.init.constant_(m.bias, 0) 
       elif isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       elif isinstance(m, (nn.BatchNorm2d,nn.LayerNorm)):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)

    def forward(self, pre_data, post_data):
        #####################
        # encoder
        #####################
        input_data = torch.cat([pre_data, post_data],dim=1)
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        #####################
        # decoder
        #####################
        up_feature_5 = self.up_sample_1(down_feature_4)
        concat_feature_5 = torch.cat([up_feature_5, feature_4], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, feature_3], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, feature_2], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, feature_1], dim=1)
        p1 = self.classifier1(concat_feature_8)
        return p1
        # output = F.softmax(output_feature, dim=1)
        # return output_feature, output


class HRSCD1(nn.Module):
    def __init__(self, in_dim=3, nc=7):
        super(HRSCD1, self).__init__()

        self.conv_block_1 = nn.Sequential(
            BasicBlock(inplanes=in_dim, planes=16, dilation=1, use_1x1=True),
            BasicBlock(inplanes=16, planes=16, dilation=1),
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            BasicBlock(inplanes=16, planes=32, dilation=1, use_1x1=True),
            BasicBlock(inplanes=32, planes=32, dilation=1),
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            BasicBlock(inplanes=32, planes=64, dilation=1, use_1x1=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
            BasicBlock(inplanes=64, planes=64, dilation=1),
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            BasicBlock(inplanes=64, planes=128, dilation=1, use_1x1=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
            BasicBlock(inplanes=128, planes=128, dilation=1),
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
        )
        self.conv_block_5 = nn.Sequential(
            BasicBlock(inplanes=256, planes=128, dilation=1, use_1x1=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
            BasicBlock(inplanes=128, planes=64, dilation=1, use_1x1=True),
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
        )
        self.conv_block_6 = nn.Sequential(
            BasicBlock(inplanes=128, planes=64, dilation=1, use_1x1=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
            BasicBlock(inplanes=64, planes=32, dilation=1, use_1x1=True),
        )

        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=32, planes=32, dilation=1),
        )
        self.conv_block_7 = nn.Sequential(
            BasicBlock(inplanes=64, planes=32, dilation=1, use_1x1=True),
            BasicBlock(inplanes=32, planes=16, dilation=1, use_1x1=True),
        )

        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=16, planes=16, dilation=1),
        )
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=nc, kernel_size=1, padding=0),
        )
        self.apply(self._init_weights)

    def _init_weights(self,m):
       if isinstance(m, nn.Linear):
          nn.init.xavier_normal_(m.weight)
          nn.init.constant_(m.bias, 0) 
       elif isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       elif isinstance(m, (nn.BatchNorm2d,nn.LayerNorm)):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def decoder(self, down_feature_4, feature_4, feature_3, feature_2, feature_1):
        up_feature_5 = self.up_sample_1(down_feature_4)
        concat_feature_5 = torch.cat([up_feature_5, feature_4], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, feature_3], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, feature_2], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, feature_1], dim=1)
        return concat_feature_8

    def forward(self,  pre_data, post_data):
        #####################
        # encoder
        #####################
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)

        #####################
        # decoder
        #####################
        p1 = self.decoder(down_feature_41, feature_41, feature_31, feature_21, feature_11)
        p2 = self.decoder(down_feature_42, feature_42, feature_32, feature_22, feature_12)
        
        p1 = self.classifier1(p1)
        p2 = self.classifier1(p2)
        cd = torch.argmax(p1, dim=1) != torch.argmax(p2, dim=1)

        return p1, p2, cd

class HRSCD3(nn.Module):
    def __init__(self, in_dim=3, nc=7):
        super(HRSCD3, self).__init__()
        self.seg = HRSCD1(in_dim=in_dim,nc=nc)
        self.cd = HRSCD2(in_dim=in_dim,nc=2)
    
    def forward(self, pre_data, post_data):
        p1, p2, _ = self.seg(pre_data, post_data)
        cd = self.cd(pre_data, post_data)
        cd = torch.argmax(cd, dim=1)
        return p1,p2,cd
        
class HRSCD4(nn.Module):
    def __init__(self, in_dim=3, nc=7):
        super(HRSCD4, self).__init__()
        self.seg_conv_block_1 = nn.Sequential(
            BasicBlock(inplanes=in_dim, planes=16, dilation=1, use_1x1=True),
            BasicBlock(inplanes=16, planes=16, dilation=1),
        )
        self.cd_conv_block_1 = nn.Sequential(
            BasicBlock(inplanes=in_dim*2, planes=16, dilation=1, use_1x1=True),
            BasicBlock(inplanes=16, planes=16, dilation=1),
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.seg_conv_block_2 = nn.Sequential(
            BasicBlock(inplanes=16, planes=32, dilation=1, use_1x1=True),
            BasicBlock(inplanes=32, planes=32, dilation=1),
        )
        self.cd_conv_block_2 = nn.Sequential(
            BasicBlock(inplanes=16, planes=32, dilation=1, use_1x1=True),
            BasicBlock(inplanes=32, planes=32, dilation=1),
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.seg_conv_block_3 = nn.Sequential(
            BasicBlock(inplanes=32, planes=64, dilation=1, use_1x1=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
            BasicBlock(inplanes=64, planes=64, dilation=1),
        )
        self.cd_conv_block_3 = nn.Sequential(
            BasicBlock(inplanes=32, planes=64, dilation=1, use_1x1=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
            BasicBlock(inplanes=64, planes=64, dilation=1),
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.seg_conv_block_4 = nn.Sequential(
            BasicBlock(inplanes=64, planes=128, dilation=1, use_1x1=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
            BasicBlock(inplanes=128, planes=128, dilation=1),
        )
        self.cd_conv_block_4 = nn.Sequential(
            BasicBlock(inplanes=64, planes=128, dilation=1, use_1x1=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
            BasicBlock(inplanes=128, planes=128, dilation=1),
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.seg_up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
        )
        self.cd_up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
        )
        self.seg_conv_block_5 = nn.Sequential(
            BasicBlock(inplanes=256, planes=128, dilation=1, use_1x1=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
            BasicBlock(inplanes=128, planes=64, dilation=1, use_1x1=True),
        )
        self.cd_conv_block_5 = nn.Sequential(
            BasicBlock(inplanes=512, planes=128, dilation=1, use_1x1=True),
            BasicBlock(inplanes=128, planes=128, dilation=1),
            BasicBlock(inplanes=128, planes=64, dilation=1, use_1x1=True),
        )

        self.seg_up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
        )
        self.cd_up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
        )
        self.seg_conv_block_6 = nn.Sequential(
            BasicBlock(inplanes=128, planes=64, dilation=1, use_1x1=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
            BasicBlock(inplanes=64, planes=32, dilation=1, use_1x1=True),
        )
        self.cd_conv_block_6 = nn.Sequential(
            BasicBlock(inplanes=256, planes=64, dilation=1, use_1x1=True),
            BasicBlock(inplanes=64, planes=64, dilation=1),
            BasicBlock(inplanes=64, planes=32, dilation=1, use_1x1=True),
        )

        self.seg_up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=32, planes=32, dilation=1),
        )
        self.cd_up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=32, planes=32, dilation=1),
        )
        self.seg_conv_block_7 = nn.Sequential(
            BasicBlock(inplanes=64, planes=32, dilation=1, use_1x1=True),
            BasicBlock(inplanes=32, planes=16, dilation=1, use_1x1=True),
        )
        self.cd_conv_block_7 = nn.Sequential(
            BasicBlock(inplanes=128, planes=32, dilation=1, use_1x1=True),
            BasicBlock(inplanes=32, planes=16, dilation=1, use_1x1=True),
        )
        self.seg_up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=16, planes=16, dilation=1),
        )
        self.cd_up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            BasicBlock(inplanes=16, planes=16, dilation=1),
        )
        self.seg_classifier1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=nc, kernel_size=1, padding=0),
        )
        self.cd_classifier1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, padding=0),
        )

        

    def seg_encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.seg_conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.seg_conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.seg_conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.seg_conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def cd_encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.cd_conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.cd_conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.cd_conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.cd_conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def seg_decoder(self, down_feature_4, feature_4, feature_3, feature_2, feature_1):
        up_feature_5 = self.seg_up_sample_1(down_feature_4)
        concat_feature_5 = torch.cat([up_feature_5, feature_4], dim=1)
        feature_5 = self.seg_conv_block_5(concat_feature_5)

        up_feature_6 = self.seg_up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, feature_3], dim=1)
        feature_6 = self.seg_conv_block_6(concat_feature_6)

        up_feature_7 = self.seg_up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, feature_2], dim=1)
        feature_7 = self.seg_conv_block_7(concat_feature_7)

        up_feature_8 = self.seg_up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, feature_1], dim=1)
        return concat_feature_8

    def cd_decoder(self, feature_41, feature_31, feature_21, feature_11,
                        feature_42, feature_32, feature_22, feature_12,
                        down_feature_4, feature_4, feature_3, feature_2, feature_1):
        up_feature_5 = self.cd_up_sample_1(down_feature_4)
        concat_feature_5 = torch.cat([up_feature_5, feature_4,feature_41, feature_42], dim=1)
        feature_5 = self.cd_conv_block_5(concat_feature_5)

        up_feature_6 = self.cd_up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, feature_3,feature_31,feature_32], dim=1)
        feature_6 = self.cd_conv_block_6(concat_feature_6)

        up_feature_7 = self.cd_up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, feature_2,feature_21,feature_22], dim=1)
        feature_7 = self.cd_conv_block_7(concat_feature_7)

        up_feature_8 = self.cd_up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, feature_1,feature_11,feature_12], dim=1)
        return concat_feature_8
        return

    def forward(self,  pre_data, post_data):
        #####################
        # encoder
        #####################
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.seg_encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.seg_encoder(post_data)
        down_feature_4, feature_4, feature_3, feature_2, feature_1= self.cd_encoder(torch.cat([pre_data, post_data], dim=1))

        #####################
        # decoder
        #####################
        p1 = self.seg_decoder(down_feature_41, feature_41, feature_31, feature_21, feature_11)
        p2 = self.seg_decoder(down_feature_42, feature_42, feature_32, feature_22, feature_12)
        
        p1 = self.seg_classifier1(p1)
        p2 = self.seg_classifier1(p2)
        cd = self.cd_decoder(feature_41, feature_31, feature_21, feature_11,
                             feature_42, feature_32, feature_22, feature_12,
                             down_feature_4, feature_4, feature_3, feature_2, feature_1)
        cd = self.cd_classifier1(cd)
        cd = torch.argmax(cd, dim=1)
        return p1, p2, cd
if __name__ == '__main__':
    x1 = torch.rand(5,3,256,256)
    x2 = torch.rand(5,3,256,256)
    net = HRSCD1(in_dim=3)
    p1,p2,cd = net(x1,x2)
    print(p1.shape, p2.shape, cd.shape)
    net = HRSCD2(in_dim=3)
    cd = net(x1,x2)
    print(cd.shape)
    net = HRSCD3()
    p1,p2,cd = net(x1,x2)
    print(p1.shape, p2.shape, cd.shape)
    net = HRSCD4()
    p1,p2,cd = net(x1,x2)
    print(p1.shape, p2.shape, cd.shape)
    