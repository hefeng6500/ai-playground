# -*- coding: utf-8 -*-
"""
随机森林分类器 Demo
演示如何使用随机森林进行分类任务、特征重要性分析、超参数调优等
"""

from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import sys

# 确保输出使用 UTF-8 编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang HK', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("随机森林分类器 Demo")
print("=" * 70)

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

print(f"   数据集形状: {X.shape}")
print(f"   标签形状: {y.shape}")
print(f"   类别数量: {len(np.unique(y))}")
print(f"   特征名称: {[feature_names_cn[name] for name in feature_names]}")

# 2. 数据划分
print("\n2. 数据划分（训练集:测试集 = 7:3）...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   训练集大小: {X_train.shape[0]}")
print(f"   测试集大小: {X_test.shape[0]}")

# 3. 基础随机森林模型
print("\n3. 基础随机森林模型（n_estimators=100）...")
rf_basic = RandomForestClassifier(n_estimators=100, random_state=42)
rf_basic.fit(X_train, y_train)

y_pred_basic = rf_basic.predict(X_test)
accuracy_basic = accuracy_score(y_test, y_pred_basic)
print(f"   训练集准确率: {rf_basic.score(X_train, y_train):.4f}")
print(f"   测试集准确率: {accuracy_basic:.4f}")

# 4. 交叉验证评估
print("\n4. 交叉验证评估（5折）...")
cv_scores = cross_val_score(rf_basic, X, y, cv=5, scoring='accuracy')
print(f"   各折准确率: {[f'{score:.4f}' for score in cv_scores]}")
print(f"   平均准确率: {cv_scores.mean():.4f}")
print(f"   标准差: {cv_scores.std():.4f}")

# 5. 详细分类报告
print("\n5. 分类评估报告...")
print(classification_report(y_test, y_pred_basic, target_names=target_names))

# 6. 混淆矩阵
print("\n6. 混淆矩阵...")
cm = confusion_matrix(y_test, y_pred_basic)
print(cm)

# 7. 特征重要性分析
print("\n7. 特征重要性分析...")
importances = rf_basic.feature_importances_
indices = np.argsort(importances)[::-1]
print("   特征重要性排序:")
for i in range(len(indices)):
    print(f"   {i+1}. {feature_names_cn[feature_names[indices[i]]]}: {importances[indices[i]]:.4f}")

# 8. 超参数调优
print("\n8. 超参数调优（GridSearchCV）...")
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("   搜索空间大小:", len(param_grid['n_estimators']) * 
      len(param_grid['max_depth']) * 
      len(param_grid['min_samples_split']) * 
      len(param_grid['min_samples_leaf']), "个组合")
print("   正在进行网格搜索（可能需要一段时间）...")

rf_grid = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf_grid, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n   最优超参数: {grid_search.best_params_}")
print(f"   最优交叉验证得分: {grid_search.best_score_:.4f}")

# 9. 最优模型评估
print("\n9. 最优模型评估...")
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"   训练集准确率: {best_rf.score(X_train, y_train):.4f}")
print(f"   测试集准确率: {accuracy_best:.4f}")
print(f"   相比基础模型的改进: {(accuracy_best - accuracy_basic):.4f}")

# 10. 不同树数量对性能的影响
print("\n10. 分析不同树数量对性能的影响...")
n_estimators_range = [10, 20, 50, 100, 150, 200, 250, 300]
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf_temp.fit(X_train, y_train)
    train_scores.append(rf_temp.score(X_train, y_train))
    test_scores.append(rf_temp.score(X_test, y_test))

print("   树数量 | 训练准确率 | 测试准确率")
for n_est, train_score, test_score in zip(n_estimators_range, train_scores, test_scores):
    print(f"   {n_est:>3d}   |  {train_score:.4f}   |   {test_score:.4f}")

# 11. 可视化
print("\n11. 生成可视化图表...")

fig = plt.figure(figsize=(15, 12))

# 特征重要性柱状图
ax1 = plt.subplot(2, 2, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
bars = ax1.bar(range(len(indices)), importances[indices], color=colors)
ax1.set_xlabel('特征', fontsize=11, fontweight='bold')
ax1.set_ylabel('重要性', fontsize=11, fontweight='bold')
ax1.set_title('特征重要性分析', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(indices)))
ax1.set_xticklabels([feature_names_cn[feature_names[i]] for i in indices], 
                     rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# 混淆矩阵热力图
ax2 = plt.subplot(2, 2, 2)
im = ax2.imshow(cm, cmap='Blues', aspect='auto')
ax2.set_xlabel('预测类别', fontsize=11, fontweight='bold')
ax2.set_ylabel('真实类别', fontsize=11, fontweight='bold')
ax2.set_title('混淆矩阵 - 最优模型', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(target_names)))
ax2.set_yticks(range(len(target_names)))
ax2.set_xticklabels(target_names)
ax2.set_yticklabels(target_names)

# 添加数值标签
for i in range(len(target_names)):
    for j in range(len(target_names)):
        text = ax2.text(j, i, cm[i, j],
                       ha="center", va="center", color="black", fontweight='bold')
plt.colorbar(im, ax=ax2)

# 树数量对性能的影响
ax3 = plt.subplot(2, 2, 3)
ax3.plot(n_estimators_range, train_scores, 'o-', label='训练集', linewidth=2, markersize=6)
ax3.plot(n_estimators_range, test_scores, 's-', label='测试集', linewidth=2, markersize=6)
ax3.set_xlabel('树的数量', fontsize=11, fontweight='bold')
ax3.set_ylabel('准确率', fontsize=11, fontweight='bold')
ax3.set_title('树数量对性能的影响', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 交叉验证得分分布
ax4 = plt.subplot(2, 2, 4)
ax4.boxplot([cv_scores], labels=['5折交叉验证'], patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax4.scatter([1] * len(cv_scores), cv_scores, color='red', s=50, alpha=0.6, label='各折得分')
ax4.set_ylabel('准确率', fontsize=11, fontweight='bold')
ax4.set_title('交叉验证得分分布', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('random_forest_analysis.png', dpi=150, bbox_inches='tight')
print("   ✓ 可视化图表已保存为: random_forest_analysis.png")
plt.show()

# 12. 模型参数总结
print("\n12. 模型参数总结")
print("   基础模型参数:")
print(f"   - n_estimators: 100")
print(f"   - max_depth: None (无限制)")
print(f"   - min_samples_split: 2")
print(f"   - min_samples_leaf: 1")
print(f"\n   最优模型参数:")
for key, value in grid_search.best_params_.items():
    print(f"   - {key}: {value}")

print("\n" + "=" * 70)
print("Demo 完成!")
print("=" * 70)

