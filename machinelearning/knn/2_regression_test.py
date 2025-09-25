#回归任务
from sklearn.neighbors import KNeighborsRegressor
X = [[2, 1], [3, 1], [1, 4], [2, 6]]
y = [0.5, 1.5, 4, 3.2]

# 创建模型
knn = KNeighborsRegressor(n_neighbors=2)

# 训练
knn.fit(X, y)

print(knn.predict([[4, 9]]))