import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Define the ResNeXt block
class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality, stride=1):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                 groups=cardinality)
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1x1_1(x)
        out = F.relu(out)
        out = self.conv3x3(out)
        out = F.relu(out)
        out = self.conv1x1_2(out)
        out = self.bn(out)
        out += residual
        return out


# Define the ResNeXt architecture
class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, num_classes=2):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(cardinality, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(cardinality, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(cardinality, num_blocks[2], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, cardinality, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNeXtBlock(self.in_channels, 128, cardinality, stride))
            self.in_channels = 128 * cardinality
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def _resnext(num_blocks, cardinality, **kwargs):
    model = ResNeXt(num_blocks, cardinality, **kwargs)
    return model

def resnext18(**kwargs):
    return _resnext(num_blocks=[2,2,2], cardinality=16, **kwargs)
