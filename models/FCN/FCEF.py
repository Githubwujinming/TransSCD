'''
R. Caye Daudt, B. Le Saux, and A. Boulch, “Fully convolutional siamese networks for change detection,” in Proceedings - International Conference on Image Processing, ICIP, 2018, pp. 4063–4067.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCEF(nn.Module):
    def __init__(self, in_dim, nc=7):
        super(FCEF, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, padding=0),
        )
        self.classifier1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=nc, kernel_size=1, padding=0),
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
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
        cd = self.conv_block_8(concat_feature_8)
        p1 = self.classifier1(concat_feature_8)
        p2 = self.classifier2(concat_feature_8)
        return p1, p2, cd
        # output = F.softmax(output_feature, dim=1)
        # return output_feature, output
