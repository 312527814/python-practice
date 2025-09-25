#特征工程-归一化
from sklearn.preprocessing import MinMaxScaler
X = [[2, 1], [3, 1], [1, 4], [2, 6]]

scaler = MinMaxScaler(feature_range=(-1, 1))

X_scaled = scaler.fit_transform(X)

print(X_scaled)