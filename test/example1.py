import numpy as np
import matplotlib.pyplot as plt

# 创建二维数组
# X_2d = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
# X_2d = np.array([[1], [2], [3], [4], [5]])
X_2d = np.array([1, 2, 3, 4, 5])
y_2d = np.array([10, 20, 30, 40, 50])

# 使用二维数组的第一列作为x
fig, ax = plt.subplots()
ax.scatter(X_2d, y_2d, c='purple')
ax.set_title('使用二维数组的列数据')
plt.show()