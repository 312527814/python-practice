import numpy as np

# 1. 阶跃函数
# x传入标量
def step_function0(x):
    if x >= 0:
        return 1
    else:
        return 0

# x传入向量或矩阵
def step_function(x):
    return np.array(x >= 0, dtype=int)

# 2. Sigmoid 函数
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# 3. ReLU 函数
def relu(x):
    return np.maximum(0, x)

# 4. Softmax函数
# 输入x为向量
def softmax0(x):
    # 溢出对策
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 输入x为矩阵 N×C
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)    # 减去每条数据中xi的最大值
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # 溢出对策
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 5. 恒等函数
def identity(x):
    return x

# 测试
x = np.array([[0, 1, 2], [3, 4, 5], [-1, -2, -3], [-6, -4, -5]])

# print(softmax(x))