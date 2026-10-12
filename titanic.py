# 导入需要的库
import pandas as pd
import numpy as np

# 读取csv数据集
df = pd.read_csv("train.csv")

# ========== 查看数据 ==========
# 看前5行数据
print(df.head())

# 查看数据集基本信息（列名、数据类型、缺失值数量）
print("\n===== 数据基本信息 =====")
print(df.info())

# 数值列的统计描述（均值、最大最小值、标准差等）
print("\n===== 统计信息 =====")
print(df.describe())

# 统计每一列缺失值数量
print("\n===== 每列缺失值统计 =====")
print(df.isnull().sum())

# 1. Age：用年龄平均值填充缺失
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 2. Embarked：只有2个缺失，用出现最多的港口填充（众数）
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. Cabin缺失太多，直接删掉这一列（缺失率太高，信息没用）
df = df.drop('Cabin', axis=1)

# 填充完，再次打印检查缺失值，确认全部清理完成
print("\n===== 清洗后，每列缺失值统计 =====")
print(df.isnull().sum())

# 按性别统计生还情况
print("\n===== 不同性别的生还统计 =====")
print(df.groupby('Sex')['Survived'].mean())

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常
# 性别生还率柱状图
sex_survive = df.groupby('Sex')['Survived'].mean()
sex_survive.plot(kind='bar')
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.show()

# 按船舱等级Pclass统计生还率
print("\n===== 不同舱位的生还统计 =====")
pclass_survive = df.groupby('Pclass')['Survived'].mean()
print(pclass_survive)

# 画舱位生还率柱状图
pclass_survive.plot(kind='bar', color='orange')
plt.title('Survival Rate by Pclass')
plt.ylabel('Survival Rate')
plt.xlabel('Pclass(1=一等舱,2=二等舱,3=三等舱)')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 性别转为0/1数字，模型只能识别数字
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

# 选取特征和预测目标
X = df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y = df['Survived']

# 划分训练集、测试集，8:2拆分
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 构建并训练逻辑回归模型
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

# 模型预测 + 计算准确率
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(f"\n模型预测准确率：{acc:.2f}")
