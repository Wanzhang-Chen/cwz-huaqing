import torch
from torch import nn


# 定义残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 用于匹配通道数/尺寸

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# 定义 ResNet 网络
class ResNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetSmall, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 输出: 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(64, 2, stride=1)   # 输出: 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 输出: 16x16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 输出: 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 输出: 4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: 1x1
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # 如果通道数不同或需要下采样
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# 测试模型
if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)  # CIFAR10 输入
    model = ResNetSmall(num_classes=10)
    y = model(x)
    print(y.shape)  # 应输出: torch.Size([1, 10])
