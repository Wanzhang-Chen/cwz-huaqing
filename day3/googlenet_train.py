# train.py
import time
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from googlenet import GoogLeNet

# 使用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# CIFAR-10 数据加载（路径改为 dataset_chen）
transform = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.CIFAR10(root="./dataset_chen", train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset_chen", train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 初始化模型
model = GoogLeNet(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 日志记录
writer = SummaryWriter("logs_google")
os.makedirs("model_save", exist_ok=True)

# 训练设置
epochs = 10
total_train_step = 0
total_test_step = 0
start_time = time.time()

for epoch in range(epochs):
    print(f"----- 第 {epoch+1} 轮训练开始 -----")
    model.train()
    for imgs, targets in train_loader:
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

    # 测试模型
    model.eval()
    total_test_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            total_correct += (outputs.argmax(1) == targets).sum().item()

    accuracy = total_correct / len(test_data)
    print(f"[测试] Loss: {total_test_loss:.4f} | Accuracy: {accuracy:.4f}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(model.state_dict(), f"model_save/googlenet_{epoch}.pth")
    print("模型已保存")

writer.close()

'''
----- 第 10 轮训练开始 -----
[训练 step 7500] Loss: 1.1051
[测试] Loss: 177.5889 | Accuracy: 0.5858
模型已保存
'''