import torch 
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 第一步：准备数据 (Data Preparation)
# ==========================================

# 1. 生成 X 数据
# torch.linspace(-10, 10, 100): 在 -10 到 10 之间均匀切出 100 个点
X = torch.linspace(-10, 10, 100) 

# 2. 升维 (非常关键！)
# 原始 X 的形状是 [100] (一维向量)。
# 但 PyTorch 的矩阵乘法通常要求是 [样本数, 特征数] 的二维矩阵。
# unsqueeze(1) 把它变成了 [100, 1]，表示 100 个样本，每个样本有 1 个特征(x坐标)。
X = X.unsqueeze(1) 

# 3. 生成噪音
# torch.randn(100, 1): 生成 100行1列 的随机数，服从标准正态分布(均值0，方差1)
noise = torch.randn(100, 1)

# 4. 生成真实标签 y
# y = 3x + 2 + 噪音。
# 我们希望 AI 能从这些杂乱的点中，学出 w=3, b=2 这个规律。
y = 3 * X + 2 + noise * 0.5

# 5. 画图看看
# .numpy(): 重点！Matplotlib 不认识 PyTorch 的 Tensor，必须转成 NumPy 数组才能画图。
plt.scatter(X.numpy(), y.numpy())
plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征数=1，输出特征数=1

    def forward(self, x):
        return self.linear(x)
    
mmodel = LinearRegressionModel()


criterion = nn.MSELoss()

optimnizer = optim.SGD(mmodel.parameters(), lr=0.01)

epochs = 10

for epoch in range(epochs):
    predicted = mmodel(X)
    loss = criterion(predicted, y)
    
    optimnizer.zero_grad()
    
    loss.backward()
    
    optimnizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    
predicted = mmodel(X)

plt.scatter(X.numpy(), y.numpy(),label='Original Data')
plt.plot(X.numpy(), predicted.detach().numpy(),label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()