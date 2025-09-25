#加载模型

import joblib   # 用来保存对象到文件和加载对象

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV   # 划分数据集，网格搜索交叉验证
from sklearn.compose import ColumnTransformer   # 列转换，做特征转换
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler    # 独热编码和标准化

# 1. 加载数据集
heart_disease_data = pd.read_csv("../data/heart_disease.csv")

# 数据清洗
heart_disease_data.dropna(inplace=True)

heart_disease_data.info()
# print(heart_disease_data.head())

# 2. 数据集划分
# 定义特征
X = heart_disease_data.drop("是否患有心脏病", axis=1)
y = heart_disease_data["是否患有心脏病"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程
# 数值型特征
numerical_features = ["年龄","静息血压","胆固醇","最大心率","运动后的ST下降","主血管数量"]
# 类别型特征（多元分类）
categorical_features = ["胸痛类型","静息心电图结果","峰值ST段的斜率","地中海贫血"]
# 二元类别特征
binary_features = ["性别","空腹血糖","运动性心绞痛"]

# 创建列转换器
transformer = ColumnTransformer(
    # (名称，操作，特征列表)
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("bin", "passthrough", binary_features)
    ])

# 执行特征转换
x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

model = joblib.load("knn_heart_disease.joblib")


data= x_test[100:101]
print(data)
y_pred = model.predict(data)
print(y_pred, y_test.iloc[100])


