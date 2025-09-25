import pandas as pd
# import seaborn as sns
from sklearn.metrics import confusion_matrix
labels = ["猫", "狗"]  # 分类标签

y_true = ["猫", "猫", "猫", "猫", "猫", "猫", "狗", "狗", "狗", "狗"]  # 真实值
y_pred = ["猫", "猫", "狗", "猫", "猫", "猫", "猫", "猫", "狗", "狗"]  # 预测值

matrix = confusion_matrix(y_true, y_pred, labels=labels)

print(pd.DataFrame(matrix, columns=labels, index=labels))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)

print(accuracy)


import pandas as pd
from sklearn.metrics import  precision_score

# 多分类精确率（每个类）
precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None)

print(precision_per_class)

