import torch

# 1. 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("✅ 恭喜！PyTorch 成功检测到 GPU！")
    print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA版本: {torch.version.cuda}")
    
    # 2. 做一个简单的矩阵运算测试算力
    x = torch.tensor([1.0, 2.0]).cuda()
    print(f"   测试张量已存入显存: {x}")
else:
    print("❌ 警告：PyTorch 目前只能使用 CPU，未检测到 GPU。")