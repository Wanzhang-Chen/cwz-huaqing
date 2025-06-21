import torch
from torch import nn
import torch.nn.functional as F


# 简化版 MLP，用于 MogaBlock 内部
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super(MLP, self).__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# 简化版 MogaBlock 模块（可堆叠）
class MogaBlock(nn.Module):
    def __init__(self, dim):
        super(MogaBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.dwconv(x)

        # BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.mlp(x)
        # BHWC -> BCHW
        x = x.permute(0, 3, 1, 2)

        return x + residual


# MogaNet 类似结构（模仿 AlexNet 的格式）
class moganet(nn.Module):
    def __init__(self, num_classes=10):
        super(moganet, self).__init__()
        self.model = nn.Sequential(
            # Stem 层：初步特征提取
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输出: 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16x16

            # MogaBlock Stage 1
            MogaBlock(32),
            nn.Conv2d(32, 64, kernel_size=1),  # 升通道
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 8x8

            # MogaBlock Stage 2
            MogaBlock(64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 4x4

            # MogaBlock Stage 3
            MogaBlock(128),
            nn.ReLU(inplace=True),

            nn.Flatten(),  # 展平：128×4×4
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# 测试 MogaNet 模型
if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)  # 输入 CIFAR-10 样本
    model = moganet()
    y = model(x)
    print(y.shape)  # 应输出 torch.Size([1, 10])

'''
-----第10轮训练开始-----
第7500的训练的loss:1.9463801383972168
训练时间1875.2992072105408
整体测试集上的loss:294.4260901212692
整体测试集上的正确率：0.6052000105381012
模型已保存
'''