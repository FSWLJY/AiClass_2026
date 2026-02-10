import torch
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


# ==========================================
# 第二步：初始化模型参数 (Model Initialization)
# ==========================================

# randn(1): 随机生成一个初始权重 w。
# requires_grad=True: 【全场最核心】
# 这告诉 PyTorch：“请盯着这个变量！后续所有涉及它的计算，都要记在小本本上，
# 方便一会儿自动求导算梯度”。如果不加这个，就无法反向传播。
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# .item(): 把 Tensor 里唯一的那个数字拿出来，变成普通的 Python float，打印时好看。
print(f'初始随机参数: w = {w.item():.4f}, b = {b.item():.4f}')


# ==========================================
# 第三步：定义函数 (Functions)
# ==========================================

# 前向传播函数：根据当前的 w 和 b，计算预测值
def forward(x):
    return x * w + b  # 矩阵乘法/广播机制

# 损失函数：计算 预测值 和 真实值 差得有多远
# MSE (Mean Squared Error): 均方误差
def mse_loss(predicted, target):
    return ((predicted - target) ** 2).mean()


# ==========================================
# 第四步：训练循环 (Training Loop)
# ==========================================

learning_rate = 0.01  # 学习率：每次改错改多大劲
epochs = 20           # 训练轮数：把数据学几遍

for epoch in range(epochs):
    
    # 1. 前向传播 (Forward) —— "猜"
    predicted = forward(X)
    
    # 2. 计算损失 (Loss) —— "对答案"
    loss = mse_loss(predicted, y)
    
    # 3. 反向传播 (Backward) —— "找锅"
    # 这一行代码执行后，PyTorch 会自动算出 loss 对 w 和 b 的导数，
    # 并分别存放在 w.grad 和 b.grad 里面。
    loss.backward()
    
    # 4. 更新参数 (Update) —— "改错"
    # w.data: 我们只想改数值，不想让这个“修改操作”被记录到计算图中，所以用 .data
    # -= : 沿着梯度的反方向走（梯度下降）
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data
    
    # 5. 清空梯度 (Zero Grad) —— "擦黑板"
    # 【非常重要】PyTorch 的梯度默认是累加的。
    # 如果不清零，下一次 backward 算出的梯度会和这一次的加在一起，就乱套了。
    w.grad.data.zero_()
    b.grad.data.zero_()
    
    # 打印进度
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}')
    
    # ================= 动态画图部分 =================
    if epoch % 2 == 0: # 每2轮画一次，防止图太多
        # .detach(): 【难点】
        # predicted 是带有梯度信息的 Tensor。Matplotlib 不需要梯度，
        # detach() 把它从计算图中“切断”，变成一个普通的 Tensor，再 .numpy() 转数组。
        curr_pred = predicted.detach().numpy()
        
        plt.scatter(X.numpy(), y.numpy(), label='Original Data', alpha=0.5)
        plt.plot(X.numpy(), curr_pred, color='red', label='Fitted Line')
        plt.title(f'Epoch {epoch+1}: y = {w.item():.2f}x + {b.item():.2f}')
        plt.legend()
        plt.show()

print(f'真实参数: w = 3, b = 2')