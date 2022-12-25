import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x = x + shortcut
        x = F.relu(x)

        return x


class Resnet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super(Resnet, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.in_batch = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_block(64, 64, 1)
        self.layer2 = self.make_block(64, 128, 2)
        self.layer3 = self.make_block(128, 256, 2)
        self.layer4 = self.make_block(256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def make_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_batch(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        return x
