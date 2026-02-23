import torch
from model import BiCrack  # 从你刚刚创建的 model.py 里导入模型
from loss import BiCrackLoss  # 从你刚刚创建的 loss.py 里导入损失函数


def main():
    # 1. 检查你的电脑是否有显卡 (CUDA)，没有就用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的计算设备: {device}")

    # 2. 把模型和损失函数加载到设备上
    model = BiCrack(num_classes=1).to(device)
    criterion = BiCrackLoss(w_bce=0.5, w_dice=0.5).to(device)

    # 3. 设置优化器（就像是模型的老师，告诉它怎么纠正错误）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("\n模型已就绪，正在生成模拟数据进行测试...")

    # 4. 制造“假数据”来测试。
    # 假设我们有 2 张图片 (Batch=2)，通道数是 3 (RGB)，大小是 512x512
    # 注意：为了不让你的电脑显存爆炸，这里我把批次(Batch)设成了2。
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    # 制造对应的“假标签”（0代表背景，1代表裂缝）
    dummy_target = torch.randint(0, 2, (2, 1, 512, 512)).float().to(device)

    # 5. 前向传播：让图片通过模型，得出预测结果
    output = model(dummy_input)
    print(f"预测结果的形状: {output.shape}")
    # 正常的话应该输出: torch.Size([2, 1, 512, 512])

    # 6. 计算误差：对比预测结果和真实标签
    loss = criterion(output, dummy_target)
    print(f"计算出的误差值 (Loss): {loss.item():.4f}")

    # 7. 反向传播：根据误差更新模型的内部参数（这是AI学习的核心）
    optimizer.zero_grad()  # 清空旧的梯度
    loss.backward()  # 计算新的梯度
    optimizer.step()  # 更新参数

    print("\n恭喜！反向传播成功，模型测试通关，说明代码结构完全没有问题！🎉")


if __name__ == "__main__":
    main()