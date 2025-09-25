# 目标函数
def f(x):
    return x ** 2 + 6 * x + 4

# 梯度函数（导函数）
def gradient(x):
    return 2 * x + 6


# 用列表记录所有点的轨迹
x_list = []
y_list = []

# 1. 初始化参数(自变量)和学习率
x = 1
alpha = 0.1

# 4. 重复迭代100次
for i in range(100):
    y = f(x)
    print(f"x={x}\ty={y}")
    x_list.append(x)
    y_list.append(y)

    # 2. 计算梯度
    grad = gradient(x)

    # 3. 更新参数
    x = x - alpha * grad

# print(x_list)
# print(y_list)

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 1, 0.01)

plt.plot(x, f(x))
plt.plot(x_list, y_list, 'r')    # 画出点的移动轨迹
plt.scatter(x_list, y_list, color='red')

plt.show()