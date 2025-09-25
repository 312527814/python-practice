# %%
# %%
# 目标函数
def J(x):
    return (x ** 2 - 2) ** 2


# 梯度函数（导函数）
def gradient(x):
    return 4 * x ** 3 - 8 * x




def get_xlist(x,alpha):
    x_list = []
    y_list = []
    for i in range(100):
        y = J(x)
        # # 当目标值小于1e-30时停止迭代
        # while (y:=J(x)) > 1e-30:
        print(f"x={x}\ty={y}")
        x_list.append(x)
        y_list.append(y)

        # 2. 计算梯度
        grad = gradient(x)



        # 3. 更新参数
        x = x - alpha * grad
    return  x_list , y_list

# %%
import matplotlib.pyplot as plt
import numpy as np
# 1. 初始化参数(自变量)和学习率


# xlist,ylist=get_xlist(0.9,0.05)
#
# # %%
# x = np.arange(0.9, 1.6, 0.01)
#
#
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# ax[0].plot(x, J(x))
# ax[0].plot(xlist, ylist, 'r')  # 画出点的移动轨迹
# ax[0].scatter(xlist, ylist, color='red')

# 局部放大，去掉第一个点
# x_list = x_list[1:]
# y_list = y_list[1:]

x_list,y_list=get_xlist(1,0.05)

x_list = x_list[1:]
y_list = y_list[1:]


x = np.arange(1.399, 1.425, 0.001)
ax[1].plot(x, J(x))
ax[1].plot(x_list, y_list, 'r')  # 画出点的移动轨迹
ax[1].scatter(x_list, y_list, color='red')

plt.show()
# %%
