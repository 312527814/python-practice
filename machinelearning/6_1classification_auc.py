# y_score: 正例的概率值或置信度
#例如：
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 生成一个二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=100)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# 训练一个逻辑回归模型
model = LogisticRegression()
# model.fit(x_train, y_train)

# 预测概率值（取正类的概率）
y_pred_proba = model.predict_proba(x_test)[:, 1]

# 计算AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(auc_score)
