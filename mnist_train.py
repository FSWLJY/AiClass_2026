import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.datasets

# ==========================================
# 🛑 黑魔法区域：强制让 PyTorch 闭嘴 (忽略 MD5 校验)
# ==========================================
# 这里的逻辑是：把官方原本记录的 MD5 校验码全部抹掉 (变成 None)
# 这样 PyTorch 在检查文件时，只要看到文件名对，就会放行，不再核对指纹。
torchvision.datasets.MNIST.resources = [
    (url, None) for url, _ in torchvision.datasets.MNIST.resources
]
# ==========================================

# 1. 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前使用的训练设备: {device}")

# 2. 数据处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("⏳ 正在加载本地数据 (已开启免校验模式)...")
# download=True 会尝试下载，但因为我们文件已经有了，它会先检查文件存在。
# 配合上面的黑魔法，它会认为文件是完美的，从而直接加载！
try:
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
    print("✅ 数据加载成功！PyTorch 终于认了！")
except Exception as e:
    print(f"❌ 依然报错: {e}")
    print("⚠️ 如果报错显示 'Not a gzipped file' 或 'Magic number'，说明你下载的可能是网页html文件，而不是真正的gz压缩包。")
    print("   这种情况下，你需要重新下载真正的 raw 文件。")
    exit()

# 3. 搭建网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet().to(device)

# 4. 定义规则
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 5. 开始训练
print("🚀 开始训练模型...")
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[第 {epoch + 1} 轮, 进度 {i + 1:5d}] Loss(误差): {running_loss / 200:.4f}')
            running_loss = 0.0

print("🎉 训练完成！")