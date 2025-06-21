import torch
from torch import nn


class alex(nn.Module):
    def __init__(self, num_classes=100):
        super(alex, self).__init__()

        # 卷积+池化层（提取特征）
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # 32x32 或更大
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 降一半

            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 动态获取 flatten 的输入维度
        self.flatten_dim = self._get_flatten_size()

        # 全连接层（分类器）
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def _get_flatten_size(self):
        # 用一个假数据通过 features 计算输出的 Flatten 大小
        with torch.no_grad():
            x = torch.zeros(1, 3, 256, 256)  # 输入和你的 transform.Resize 保持一致
            x = self.features(x)
            return x.view(1, -1).size(1)  # 返回 flatten 后的长度

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# ✅ 测试
if __name__ == '__main__':
    model = alex()
    x = torch.randn(1, 3, 256, 256)  # 可输入任意尺寸
    y = model(x)
    print(y.shape)  # 输出 torch.Size([1, 100])
