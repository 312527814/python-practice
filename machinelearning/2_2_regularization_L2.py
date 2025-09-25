import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归模型，Lasso回归，Ridge回归
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
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].scatter(X, y, c='y')
ax[0,1].scatter(X, y, c='y')
ax[0,2].scatter(X, y, c='y')

# plt.show()

# 2. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 过拟合：20阶，转换成20个特征的线性拟合，复杂度过高
x_train = polynomial(x_train, 20)
x_test = polynomial(x_test, 20)


# L2正则化 —— Ridge回归
ridge = Ridge(alpha=1)

# 4. 训练模型
ridge.fit(x_train, y_train)

# 5. 测试数据预测结果，计算误差（测试误差）
y_pred_test_result = ridge.predict(x_test)
# 调用均方误差函数，传入y的真实值和预测值
test_loss3 = mean_squared_error(y_test, y_pred_test_result)
# 6. 训练数据预测结果，用来划线
y_pred_tran_result = ridge.predict( polynomial(X, 20) )
# 画出拟合曲线，并标出误差
ax[0,2].plot(X, y_pred_tran_result, 'r')
ax[0,2].text(-3, 1, f"测试集均方误差：{test_loss3:.4f}")
ax[0,2].text(-3, 1.3, "Ridge回归")
# 画出所有系数的柱状图
ax[1,2].bar(np.arange(20), ridge.coef_.reshape(-1))


plt.show()