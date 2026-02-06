import os

# 1. 获取当前在哪里
current_path = os.getcwd()
print(f"📍 当前运行位置: {current_path}")

# 2. 推算 PyTorch 想要找的路径
expected_path = os.path.join(current_path, "data", "MNIST", "raw")
print(f"🔍 PyTorch 正在寻找这个文件夹: {expected_path}")

# 3. 看看文件到底在不在
if os.path.exists(expected_path):
    print("✅ 文件夹找到了！里面的文件有：")
    files = os.listdir(expected_path)
    for f in files:
        print(f"   📄 {f}")
        
    # 4. 检查是否有必须的那个文件
    required_file = "train-images-idx3-ubyte.gz"
    if required_file in files:
        print("\n🎉 关键文件名完全正确！")
    else:
        print(f"\n❌ 缺关键文件！我们需要 '{required_file}'")
        print("   (请仔细对比上面列出的文件名，看看是不是多了 .txt 或少了 .gz)")
else:
    print("\n❌ 文件夹都没找到！")
    print(f"   请检查你的 'data' 文件夹是不是建在 {current_path} 下面？")
    print("   常见错误：建成了 data/data/MNIST 或者 data/MNIST/MNIST")