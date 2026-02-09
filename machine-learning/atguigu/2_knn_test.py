from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据（这就解决了 X 和 y 从哪来的问题）
iris = load_iris()
X, y = iris.data, iris.target

# 打印一下数据
print(X.shape)
print(y.shape)
print(X[:5])
print(y[:500])




# 2. 先拆分数据集（保护测试集不被“污染”）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 预处理：只针对训练集 fit
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 学习训练集的分布
X_test_scaled = scaler.transform(X_test)       # 使用训练集的规则处理测试集

# 4. 创建并训练模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 绘制图
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train)
# 绘制测试集的点
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, marker='x')
plt.legend(['训练集', '测试集'])
plt.title("KNN分类")
plt.show()

# 5. 预测并评分
accuracy = knn.score(X_test_scaled, y_test)
print(f"修正后的准确率: {accuracy:.4f}")