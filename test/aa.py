import numpy as np
def polynomial(X, degree):
    return np.hstack([X**i for i in range(1, degree + 1)])

list2 = [[1], [2],[3]]




# print(type(list2))

# arr1d = [i for i in range(10)]
#
#
# print(type(arr1d))

# 从列表创建一维数组
arr1d = np.array([1, 2, 3, 4, 5])
# arr2d =arr1d**2
# print(arr2d)
a= arr1d.reshape(-1,1)
arr5d =polynomial(a,5)

print(arr5d.shape)
print(arr5d)