#线性回归-正规方程法(多元)
from sklearn.linear_model import LinearRegression

# 1. 定义数据
X = [[0, 3], [1, 2], [2, 1]]
y = [0, 1, 2]


# 2. 创建线性回归模型
model = LinearRegression()

# 3. 训练模型
model.fit(X, y)

# 查看模型的系数和截距
print(model.coef_)
print(model.intercept_)

# 5. 预测
x_new = [[5,4]]
y_pred = model.predict(x_new)
print(y_pred)
