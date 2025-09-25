#线性回归-梯度下降Api
from sklearn.linear_model import SGDRegressor

# 1. 定义数据
X = [[0, 3], [1, 2], [2, 1]]
y = [0, 1, 2]


model = SGDRegressor(
    loss='squared_error',#平方差损失函数
    alpha=1e-3,#正则化系数
    learning_rate='constant',
    eta0=0.001, #学习率
    max_iter=10000,#最打迭代次数
    tol=0.0001, #The stopping criterion. If it is not None, training will stop when (loss > best_loss - tol)
)

model.fit(X, y)

print(model.coef_)
print(model.intercept_)

# 5. 预测
x_new = [[5,4]]
y_pred = model.predict(x_new)
print(y_pred)
