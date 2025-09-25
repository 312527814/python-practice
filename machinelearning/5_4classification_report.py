from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
#三分类
# 1. 生成分类数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,n_informative=3, random_state=42)

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 定义模型
model = LogisticRegression(random_state=42)  # 推荐加上 random_state

# 4. 训练
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# # 7. 多分类 AUC 计算（macro-averaged OvR）
# y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
# y_pred_proba = model.predict_proba(X_test)
#
# auc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
# print(f"\nMacro AUC (One-vs-Rest): {auc:.3f}")