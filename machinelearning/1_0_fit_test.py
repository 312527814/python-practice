import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import mean_squared_error  # 均方误差损失函数

# 配置matplotlib中全局绘图参数
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 楷体字
plt.rcParams['axes.unicode_minus'] = False

# 由一个向量x，生成 degree列（degree个特征）的矩阵，(x, x^2, ... x^degree)
def polynomial(X, degree):
    return np.hstack([X**i for i in range(1, degree + 1)])

'''
机器学习步骤：
1. 读取数据（生成数据）
2. 划分训练集和测试集
3. 定义损失函数和模型
4. 训练模型
5. 预测结果，计算误差（测试误差）
'''

# 1. 读取数据（生成数据）
# 生成随机数据，扩展成二维矩阵表示，形状(300,1)
X = np.linspace(-3, 3, 300).reshape(-1, 1)
# print(X.shape)

# 基于sinX叠加随机噪声
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

# 画出散点图
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(X, y, c='y')
ax[1].scatter(X, y, c='y')
ax[2].scatter(X, y, c='y')

# plt.show()

# 2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义模型，线性回归模型
model = LinearRegression()


## 一、欠拟合：1阶线性拟合，1个参数，复杂度过低
x_train1 = x_train
x_test1 = x_test

# 4. 训练模型
model.fit(x_train1, y_train)

# 5. 预测结果，计算误差（测试误差）
y_pred1 = model.predict(x_test)
# 调用均方误差函数，传入y的真实值和预测值
test_loss1 = mean_squared_error(y_test, y_pred1)
train_loss1 = mean_squared_error(y_train, model.predict(x_train1))

# 画出拟合曲线，并标出误差
ax[0].plot(X, model.predict(X), 'r')
ax[0].text(-3, 1, f"测试集均方误差：{test_loss1:.4f}")
ax[0].text(-3, 1.3, f"训练集均方误差：{train_loss1:.4f}")

## 二、恰好拟合：5阶，转换成5个特征的线性拟合，复杂度正好
x_train2 = polynomial(x_train, 5)
x_test2 = polynomial(x_test, 5)

# 4. 训练模型
model.fit(x_train2, y_train)

# 5. 预测结果，计算误差（测试误差）
y_pred2 = model.predict(x_test2)
# 调用均方误差函数，传入y的真实值和预测值
test_loss2 = mean_squared_error(y_test, y_pred2)
train_loss2 = mean_squared_error(y_train, model.predict(x_train2))

# 画出拟合曲线，并标出误差
ax[1].plot(X, model.predict( polynomial(X, 5) ), 'r')
ax[1].text(-3, 1, f"测试集均方误差：{test_loss2:.4f}")
ax[1].text(-3, 1.3, f"训练集均方误差：{train_loss2:.4f}")

## 三、过拟合：20阶，转换成20个特征的线性拟合，复杂度过高
x_train3 = polynomial(x_train, 20)
x_test3 = polynomial(x_test, 20)

# 4. 训练模型
model.fit(x_train3, y_train)

# 5. 预测结果，计算误差（测试误差）
y_pred3 = model.predict(x_test3)
# 调用均方误差函数，传入y的真实值和预测值
test_loss3 = mean_squared_error(y_test, y_pred3)
train_loss3 = mean_squared_error(y_train, model.predict(x_train3))

# 画出拟合曲线，并标出误差
ax[2].plot(X, model.predict( polynomial(X, 20) ), 'r')
ax[2].text(-3, 1, f"测试集均方误差：{test_loss3:.4f}")
ax[2].text(-3, 1.3, f"训练集均方误差：{train_loss3:.4f}")

print(model.coef_)
print(model.intercept_)

plt.show()