# model.py
import torch
from torch import nn
import torch.nn.functional as F

# 基础 Inception 模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1, ch3_reduce, ch3, ch5_reduce, ch5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, ch1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3_reduce, ch3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5_reduce, ch5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], 1)

# GoogLeNet 主体（适配 CIFAR-10）
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.a3 = Inception(64, 32, 48, 64, 8, 16, 16)
        self.b3 = Inception(128, 64, 64, 96, 16, 32, 32)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(224, 96, 48, 104, 8, 24, 24)
        self.b4 = Inception(248, 80, 56, 112, 12, 32, 32)
        self.c4 = Inception(256, 64, 64, 128, 16, 32, 32)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x
# 测试模型
if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)  # 使用32x32的输入
    model = GoogLeNet()
    y = model(x)
    print(y.shape)  # 应该输出torch.Size([1, 10])