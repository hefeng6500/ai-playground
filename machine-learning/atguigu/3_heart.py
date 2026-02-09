import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV    # 划分数据集，网格搜索交叉验证
from sklearn.compose import ColumnTransformer   # 列转换器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder     # 标准化转换器，独热编码器
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据集
dataset = pd.read_csv('./data/heart_disease.csv')

# 处理缺失值
dataset.dropna(inplace=True)

dataset.info()
print(dataset.head())

# 2. 数据集划分
X = dataset.drop(['是否患有心脏病'], axis=1)
y = dataset["是否患有心脏病"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程：特征转换
# 数值型特征
numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]
# 类别型特征
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]

# 创建列转换器
columnTransformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop="first"), categorical_features),
        ('bin', "passthrough", binary_features),
    ]
)

# 执行特征转换
x_train = columnTransformer.fit_transform(x_train)
x_test = columnTransformer.transform(x_test)

print(x_train.shape, x_test.shape)

# # 4. 定义模型：KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)

# 5. 模型训练
knn.fit(x_train, y_train)

# 6. 模型评估
score = knn.score(x_test, y_test)
print(score)

# 7. 模型的保存
import joblib
joblib.dump(knn, 'knn_heart_disease')

# 8. 加载模型并预测
knn_loaded = joblib.load('knn_heart_disease')
print(knn_loaded.score(x_test, y_test))

# 预测整个测试集
y_pred = knn_loaded.predict(x_test)
print(f"在 {len(y_test)} 个测试样本中，有 {sum(y_pred != y_test)} 个预测错误。")


# 绘图：可视化测试集真实值和预测值
# 为了绘图，我们使用原始（未转换）的测试集特征，这样坐标轴更具可解释性
x_test_original = X.loc[y_test.index]

# 创建一个图形
plt.figure(figsize=(12, 8))

# 绘制所有测试集样本点，按真实标签着色
# y_test == 0: 未患病 (蓝色)
# y_test == 1: 患病 (红色)
scatter = plt.scatter(x_test_original['年龄'], x_test_original['最大心率'], c=y_test, cmap='coolwarm', alpha=0.8, edgecolors='k', linewidth=0.5)

# 找出预测错误的点
misclassified_mask = y_pred != y_test
misclassified_points = x_test_original[misclassified_mask]

# 在图上突出显示预测错误的点
plt.scatter(misclassified_points['年龄'], misclassified_points['最大心率'],
            s=200, facecolors='none', edgecolors='lime', linewidths=2, label='预测错误')

# 设置图表标题和坐标轴标签
plt.title('测试集预测结果可视化 (年龄 vs 最大心率)', fontsize=16)
plt.xlabel('年龄', fontsize=12)
plt.ylabel('最大心率', fontsize=12)

# 创建并显示图例
import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', label='未患病 (真实)',
                  markerfacecolor='#3b4cc0', markersize=10),
    mlines.Line2D([0], [0], marker='o', color='w', label='患病 (真实)',
                  markerfacecolor='#dd5855', markersize=10),
    mlines.Line2D([0], [0], marker='o', color='w', label='预测错误',
                  markeredgecolor='lime', markerfacecolor='none', markersize=12, markeredgewidth=2)
]
plt.legend(handles=legend_elements, title="图例")

plt.grid(linestyle='--', alpha=0.6)
plt.show()

