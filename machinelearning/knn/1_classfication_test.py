#分类任务
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 定义数据
X = np.array([[2, 1, 0], [3, 1, 0], [1, 4, 0], [2, 6, 0]])
y = np.array([0, 0, 0, 1])  # 二分类

# 定义模型
knn = KNeighborsClassifier(n_neighbors=2, weights='distance')

# 训练模型
knn.fit(X, y)


# 对新数据进行预测
print(knn.predict([[4, 9, 0]]))
