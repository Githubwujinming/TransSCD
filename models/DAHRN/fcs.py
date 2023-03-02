
import torch 
import torch.nn as nn
class FCSiamConc(nn.Module):
    def __init__(self, in_dim):
        super(FCSiamConc, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16,momentum=0.0003),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16,momentum=0.0003),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32,momentum=0.0003),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32,momentum=0.0003),
            nn.ReLU(inplace=True)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64,momentum=0.0003),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64,momentum=0.0003),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64,momentum=0.0003),
            nn.ReLU(inplace=True)
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128,momentum=0.0003),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128,momentum=0.0003),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128,momentum=0.0003),
            nn.ReLU(inplace=True)
        )

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool(feature_4)

        return down_feature_1,down_feature_2,down_feature_3,down_feature_4

    def forward(self, pre_data):
        #####################
        # decoder
        #####################
        feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        return [feature_41, feature_31, feature_21, feature_11]
