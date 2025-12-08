import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1. 准别数据
# 1.1 加载数据
data_train = pd.read_csv('../data/fashion-mnist_train.csv')
data_test = pd.read_csv('../data/fashion-mnist_test.csv')

# 1.2 将数据划分特征和目标，并转换成张量 （N，C，H，W）
X_train = torch.tensor(data_train.iloc[:, 1:].values, dtype=torch.float).reshape(-1, 1, 28, 28)
y_train = torch.tensor(data_train.iloc[:, 0].values, dtype=torch.int64)

print("第一条数据的形状：", X_train[0].shape)
print("第一条数据的像素值：", X_train[0])


# 获取第一条数据
first_image = X_train[0]
first_label = y_train[0].item()

# 将张量转换为NumPy数组并调整形状
image_np = first_image.squeeze().numpy()  # 从 [1, 28, 28] 变为 [28, 28]
print("第=======：", image_np)


# 显示图像
plt.figure(figsize=(6, 6))
plt.imshow(image_np, cmap='gray')
plt.title(f"Label: {first_label}")
plt.colorbar()
plt.show()


