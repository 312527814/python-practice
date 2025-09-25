import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score

labels = ["猫", "狗", "猪"]
y_true = ["猫", "猫", "猫", "猫", "猫", "猫", "狗", "狗", "狗", "狗", "猪", "猪", "猪"]
y_pred = ["猫", "猫", "狗", "猫", "猫", "猫", "猫", "猫", "狗", "狗", "猪", "猪", "猪"]

# 混淆矩阵
matrix = confusion_matrix(y_true, y_pred, labels=labels)
print("混淆矩阵:")
print(pd.DataFrame(matrix, columns=labels, index=labels))



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)

print(accuracy)

# 多分类精确率（每个类）
precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None)

print(precision_per_class)
precision_dict = dict(zip(labels, precision_per_class))

print(f"\n各类精确率: {precision_dict}")
print(f"猫类的精确率: {precision_dict['猫']:.3f}")