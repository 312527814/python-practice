# 特征工程-标准化
from sklearn.preprocessing import StandardScaler

X = [[2, 1, 1], [3, 1, 1], [1, 4, 1], [2, 6, 1]]
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
