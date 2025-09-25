# y_score: 正例的概率值或置信度
#例如：
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import label_binarize

# 生成三分类数据（注意参数）
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=3,
    n_informative=5,  # 满足 2^5=32 >= 3*2=6
    n_clusters_per_class=2,
    random_state=100
)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# model = LogisticRegression()
model = LogisticRegression(random_state=100)
# model.fit(x_train, y_train)

# 获取预测概率
y_pred_proba = model.predict_proba(x_test)

# 将标签二值化（one-vs-rest）
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# 计算多分类 AUC（macro 平均）
auc_score = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='macro')
print(f"Macro AUC (OvR): {auc_score:.3f}")
