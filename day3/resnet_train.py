import time
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from resnet import ResNetSmall  # 确保你已实现并正确导入该模型

# 选择设备：GPU 优先
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练集大小: {train_data_size}, 测试集大小: {test_data_size}")

# 加载数据
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 初始化模型并迁移到 GPU
model = ResNetSmall(num_classes=10).to(device)

# 损失函数 & 优化器
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 日志 & 模型保存
writer = SummaryWriter("./logs_resnet")
os.makedirs("model_save", exist_ok=True)

# 训练配置
total_train_step = 0
total_test_step = 0
epochs = 10
start_time = time.time()

for epoch in range(epochs):
    print(f"----- 第 {epoch+1} 轮训练开始 -----")
    model.train()
    for data in train_loader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"[训练 step {total_train_step}] Loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print(f"本轮训练时间: {end_time - start_time:.2f} 秒")

    # 测试阶段
    model.eval()
    total_test_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_correct += (outputs.argmax(1) == targets).sum().item()

    accuracy = total_correct / test_data_size
    print(f"[测试] Loss: {total_test_loss:.4f} | Accuracy: {accuracy:.4f}")

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy, total_test_step)
    total_test_step += 1

    torch.save(model.state_dict(), f"model_save/resnet_{epoch}.pth")
    print("模型已保存\n")

writer.close()

'''
----- 第 10 轮训练开始 -----
[训练 step 7500] Loss: 0.0918
本轮训练时间: 2732.76 秒
[测试] Loss: 114.5849 | Accuracy: 0.8111
模型已保存
'''