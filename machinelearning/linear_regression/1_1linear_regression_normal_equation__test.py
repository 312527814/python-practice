#线性回归-正规方程法
from sklearn.linear_model import LinearRegression

# 1. 定义数据
X = [[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]]
y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 58]
X = [[5], [8]]
y = [55,65]

# 2. 创建线性回归模型
model = LinearRegression()

# 3. 训练模型
model.fit(X, y)

# 查看模型的系数和截距
print(model.coef_)
print(model.intercept_)

# 5. 预测
x_new = [[5]]
y_pred = model.predict(x_new)
print(y_pred)
