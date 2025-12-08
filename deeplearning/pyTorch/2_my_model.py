import torch
from torch import nn, optim

# 自定义模型类
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=3, out_features=5)
        self.linear.weight.data = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5],
             [0.6, 0.7, 0.8, 0.9, 1.0],
             [1.1, 1.2, 1.3, 1.4, 1.5]]
        ).T
        self.linear.bias.data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    def forward(self, x):
        x = self.linear(x)
        return x

    def train(self, x, t):
        # 1. 前向传播，计算预测值
        y = self.forward(x)

        # 2. 计算损失
        loss = nn.MSELoss()
        loss_value = loss(y, t)

        # 3. 反向传播
        loss_value.backward()

        # 4. 更新参数（迭代一次）
        optimizer = optim.SGD(model.parameters(), lr=1)
        optimizer.step()

        optimizer.zero_grad()
        # 打印损失值
        print(f"Loss: {loss_value.item()}")  # 添加这行来打印损失值
        return loss_value.item()  # 也可以返回损失值

    def train(self, x, t, epochs=100):
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        losses = []  # 记录损失值

        for epoch in range(epochs):
            # 1. 前向传播，计算预测值
            y = self.forward(x)

            # 2. 计算损失
            loss_value = loss_fn(y, t)

            # 3. 反向传播
            optimizer.zero_grad()
            loss_value.backward()

            # 4. 更新参数
            optimizer.step()

            losses.append(loss_value.item())

            # 每10个epoch打印一次损失
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_value.item():.6f}')

        return losses

if __name__ == '__main__1':
    # 1. 准备数据
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    t = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float)

    # 2. 创建模型
    model = MyModel()

    # 3. 训练模型
    model.train(x, t)

    # 打印参数
    for param in model.state_dict():
        print(param)
        print(model.state_dict()[param])

if __name__ == '__main__':
    import numpy as np
    # 生成20个样本，每个样本3个特征
    np.random.seed(42)  # 设置随机种子以便复现结果
    x_data = np.random.randn(20, 3) * 2 + 1  # 均值为1，标准差为2的正态分布
    t_data = np.random.randn(20, 5) * 0.5  # 目标值，均值为0，标准差为0.5

    # 转换为PyTorch张量
    x = torch.tensor(x_data, dtype=torch.float)
    t = torch.tensor(t_data, dtype=torch.float)

    # 2. 创建模型
    model = MyModel()

    # 3. 训练模型（100个epoch）
    losses = model.train(x, t, epochs=500)

    print("\n训练完成!")
    print(f"最终损失: {losses[-1]:.6f}")

    # 打印参数
    print("\n模型参数:")
    for param in model.state_dict():
        print(f"{param}:")
        print(model.state_dict()[param])
        print()

    # 可选：绘制损失曲线
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("要绘制损失曲线，请安装matplotlib: pip install matplotlib")