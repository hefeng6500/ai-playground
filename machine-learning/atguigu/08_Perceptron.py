import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. 加载数据
# 使用经典的鸢尾花(Iris)数据集
iris = datasets.load_iris()

# 为了演示基本的线性二分类，我们只选取前两种花 (Setosa 和 Versicolor)
# 并且只选取前两个特征以便于二维图表可视化 (萼片长度和萼片宽度)
X = iris.data[:100, :2]
y = iris.target[:100]

# 2. 划分训练集和测试集
# test_size=0.3 表示 30% 的数据用于测试，70% 用于训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建感知机模型并进行训练
# max_iter: 数据集的遍历次数 (epochs)
# eta0: 学习率 (learning rate)
# random_state: 随机种子，确保每次运行结果一致
ppn = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

# 使用训练数据拟合模型
ppn.fit(X_train, y_train)

# 4. 模型评估
# 对测试集进行预测
y_pred = ppn.predict(X_test)

# 计算并打印准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"感知机模型对测试集的分类准确率: {accuracy * 100:.2f}%")

# 5. 结果可视化 (绘制决策边界)
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # 定义颜色和标记
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.matplotlib.colors.ListedColormap(colors[:len(np.unique(y))])

    # 画出决策表面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # 预测网格中每个点的类别
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # 绘制填充轮廓图
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 绘制样本散点图
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'类别 {cl}', 
                    edgecolor='black' if markers[idx] != 'x' else None)

# 设置图像支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac OSX 支持的中文黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

plt.figure(figsize=(8, 6))
# 使用所有样本数据来展示决策边界和数据分布
plot_decision_regions(X, y, classifier=ppn)

plt.xlabel('萼片长度 (cm)')
plt.ylabel('萼片宽度 (cm)')
plt.legend(loc='upper left')
plt.title('感知机 (Perceptron) 二分类示例 - 鸢尾花数据集')

# 显示图表
plt.show()
