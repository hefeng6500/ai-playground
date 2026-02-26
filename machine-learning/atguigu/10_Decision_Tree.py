# -*- coding: utf-8 -*-
"""
决策树分类器 Demo
演示如何使用决策树进行分类任务
"""

from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import sys

# 确保输出使用 UTF-8 编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang HK', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("决策树分类器 Demo")
print("=" * 60)

# 1. 加载数据集
print("\n1. 加载 Iris 数据集...")
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 中文特征名称映射
feature_names_cn = {
    'sepal length (cm)': '萼片长度 (cm)',
    'sepal width (cm)': '萼片宽度 (cm)',
    'petal length (cm)': '花瓣长度 (cm)',
    'petal width (cm)': '花瓣宽度 (cm)'
}

# 中文目标类别映射
target_names_cn = {
    'setosa': '山鸢尾',
    'versicolor': '变色鸢尾',
    'virginica': '维吉尼亚鸢尾'
}

print(f"   特征形状: {X.shape}")
print(f"   标签形状: {y.shape}")
print(f"   特征名称: {[feature_names_cn[name] for name in feature_names]}")
print(f"   目标类别: {[target_names_cn[name] for name in target_names]}")

# 2. 划分训练集和测试集
print("\n2. 划分数据集 (训练集:测试集 = 8:2)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   训练集大小: {X_train.shape[0]}")
print(f"   测试集大小: {X_test.shape[0]}")

# 3. 创建和训练决策树模型
print("\n3. 创建并训练决策树模型...")
dt_classifier = DecisionTreeClassifier(
    max_depth=4,           # 限制树的深度，防止过拟合
    min_samples_split=2,   # 分裂节点所需的最少样本数
    min_samples_leaf=1,    # 叶子节点最少样本数
    random_state=42
)
dt_classifier.fit(X_train, y_train)
print(f"   树的深度: {dt_classifier.get_depth()}")
print(f"   叶子节点数: {dt_classifier.get_n_leaves()}")

# 4. 在测试集上进行预测
print("\n4. 模型预测和评估...")
y_pred = dt_classifier.predict(X_test)

# 计算准确率
train_accuracy = dt_classifier.score(X_train, y_train)
test_accuracy = dt_classifier.score(X_test, y_test)
print(f"   训练集准确率: {train_accuracy:.4f}")
print(f"   测试集准确率: {test_accuracy:.4f}")

# 5. 详细的分类报告
print("\n5. 分类报告:")
target_names_display = [target_names_cn[name] for name in target_names]
print(classification_report(y_test, y_pred, target_names=target_names_display))

# 6. 混淆矩阵
print("\n6. 混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 7. 特征重要性
print("\n7. 特征重要性:")
feature_importance = dt_classifier.feature_importances_
for name, importance in zip(feature_names, feature_importance):
    print(f"   {feature_names_cn[name]}: {importance:.4f}")

# 8. 可视化决策树
print("\n8. 可视化决策树...")
plt.figure(figsize=(25, 10))
feature_names_display = [feature_names_cn[name] for name in feature_names]
target_names_display = [target_names_cn[name] for name in target_names]
plot_tree(
    dt_classifier,
    feature_names=feature_names_display,
    class_names=target_names_display,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("决策树可视化", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
print("   决策树已保存为 decision_tree_visualization.png")
plt.show()

# 9. 特征重要性柱状图
print("\n9. 绘制特征重要性图...")
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.title("特征重要性", fontsize=14, fontweight='bold')
plt.bar(range(X.shape[1]), feature_importance[indices])
feature_names_display = [feature_names_cn[feature_names[i]] for i in indices]
plt.xticks(range(X.shape[1]), feature_names_display, rotation=45, ha='right')
plt.ylabel("重要性")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("   特征重要性图已保存为 feature_importance.png")
plt.show()

# 10. 不同 max_depth 对模型性能的影响
print("\n10. 探索不同深度参数对模型的影响...")
train_accuracies = []
test_accuracies = []
depths = range(1, 15)

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_accuracies.append(dt.score(X_train, y_train))
    test_accuracies.append(dt.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, 'o-', label='训练集准确率', linewidth=2)
plt.plot(depths, test_accuracies, 's-', label='测试集准确率', linewidth=2)
plt.xlabel("树的最大深度 (max_depth)")
plt.ylabel("准确率")
plt.title("不同树深度对模型性能的影响", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('depth_effect.png', dpi=150, bbox_inches='tight')
print("   深度影响图已保存为 depth_effect.png")
plt.show()

print("\n" + "=" * 60)
print("Demo 完成！")
print("=" * 60)

