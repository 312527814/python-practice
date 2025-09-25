import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 生成数据
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# 2. 标准化（Lasso 对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 定义要尝试的 alpha 值
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# 4. 存储每个 alpha 的 CV 分数
cv_means = []
cv_stds = []

# 5. 手动循环：对每个 alpha 计算交叉验证得分
for alpha in alphas:
    # 创建 Lasso 模型
    lasso = Lasso(alpha=alpha, random_state=42)

    # 使用 5 折交叉验证评估性能（负均方误差）
    scores = cross_val_score(lasso, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

    # 记录平均分和标准差
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

    print(f"alpha={alpha:4.3f}, CV 负MSE = {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# 6. 找出表现最好的 alpha
cv_means = np.array(cv_means)
best_idx = np.argmax(cv_means)  # 最大负MSE（即最小 MSE）
best_alpha = alphas[best_idx]
best_score = cv_means[best_idx]

print(f"\n✅ 最优 alpha = {best_alpha}")
print(f"✅ 对应的 CV 负MSE = {best_score:.3f}")