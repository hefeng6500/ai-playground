# -*- coding: utf-8 -*-
"""
AdaBoost 分类器 Demo
演示如何使用 AdaBoost 进行分类任务、集成方式对比、超参数调优等
"""

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import numpy as np
import sys

# 确保输出使用 UTF-8 编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang HK', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("AdaBoost 分类器 Demo")
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

# 3. 基础 AdaBoost 模型
print("\n3. 基础 AdaBoost 模型（n_estimators=50）...")
ada_basic = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
ada_basic.fit(X_train, y_train)

y_pred_basic = ada_basic.predict(X_test)
accuracy_basic = accuracy_score(y_test, y_pred_basic)
print(f"   训练集准确率: {ada_basic.score(X_train, y_train):.4f}")
print(f"   测试集准确率: {accuracy_basic:.4f}")

# 4. 单个决策树模型对比
print("\n4. 单个决策树模型对比...")
dt_single = DecisionTreeClassifier(max_depth=1, random_state=42)
dt_single.fit(X_train, y_train)

y_pred_dt = dt_single.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"   决策树训练集准确率: {dt_single.score(X_train, y_train):.4f}")
print(f"   决策树测试集准确率: {accuracy_dt:.4f}")
print(f"   AdaBoost 相比单决策树的改进: {(accuracy_basic - accuracy_dt):.4f}")

# 5. 交叉验证评估
print("\n5. 交叉验证评估（5折）...")
cv_scores_ada = cross_val_score(ada_basic, X, y, cv=5, scoring='accuracy')
cv_scores_dt = cross_val_score(dt_single, X, y, cv=5, scoring='accuracy')
print(f"   AdaBoost 各折准确率: {[f'{score:.4f}' for score in cv_scores_ada]}")
print(f"   AdaBoost 平均准确率: {cv_scores_ada.mean():.4f} ± {cv_scores_ada.std():.4f}")
print(f"   决策树各折准确率: {[f'{score:.4f}' for score in cv_scores_dt]}")
print(f"   决策树平均准确率: {cv_scores_dt.mean():.4f} ± {cv_scores_dt.std():.4f}")

# 6. 详细分类报告
print("\n6. AdaBoost 分类评估报告...")
target_names_display = [target_names_cn[name] for name in target_names]
print(classification_report(y_test, y_pred_basic, target_names=target_names_display))

# 7. 混淆矩阵
print("\n7. 混淆矩阵...")
cm_ada = confusion_matrix(y_test, y_pred_basic)
print("   AdaBoost:")
print(cm_ada)

# 8. 分析弱学习器的权重
print("\n8. 分析弱学习器的权重...")
print(f"   弱学习器数量: {len(ada_basic.estimators_)}")
estimator_weights = ada_basic.estimator_weights_
estimator_errors = ada_basic.estimator_errors_
print(f"   前10个弱学习器的权重: {estimator_weights[:10]}")
print(f"   前10个弱学习器的错误率: {estimator_errors[:10]}")
print(f"   权重总和: {estimator_weights.sum():.4f}")

# 9. 超参数调优
print("\n9. 超参数调优（GridSearchCV）...")
param_grid_ada = {
    'n_estimators': [10, 30, 50, 100],
    'learning_rate': [0.5, 1.0, 1.5],
}

grid_search_ada = GridSearchCV(
    AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        random_state=42
    ),
    param_grid_ada,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search_ada.fit(X_train, y_train)
print(f"   最佳参数: {grid_search_ada.best_params_}")
print(f"   最佳交叉验证得分: {grid_search_ada.best_score_:.4f}")

ada_best = grid_search_ada.best_estimator_
y_pred_best = ada_best.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"   优化后测试集准确率: {accuracy_best:.4f}")

# 10. 不同弱学习器深度的影响
print("\n10. 不同弱学习器深度的影响...")
depths = [1, 2, 3, 4, 5]
train_scores = []
test_scores = []

for depth in depths:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=50,
        random_state=42
    )
    ada.fit(X_train, y_train)
    train_scores.append(ada.score(X_train, y_train))
    test_scores.append(ada.score(X_test, y_test))
    print(f"   深度 {depth}: 训练准确率={train_scores[-1]:.4f}, 测试准确率={test_scores[-1]:.4f}")

# 11. 使用乳腺癌数据集进行二分类演示
print("\n11. 使用乳腺癌数据集进行二分类演示...")
cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.3, random_state=42, stratify=y_cancer
)

# 标准化特征（重要）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

ada_cancer = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50,
    random_state=42
)
ada_cancer.fit(X_train_c, y_train_c)

y_pred_cancer = ada_cancer.predict(X_test_c)
accuracy_cancer = accuracy_score(y_test_c, y_pred_cancer)

# 二分类的 ROC-AUC
y_pred_proba_cancer = ada_cancer.predict_proba(X_test_c)[:, 1]
auc_score = roc_auc_score(y_test_c, y_pred_proba_cancer)

print(f"   训练集准确率: {ada_cancer.score(X_train_c, y_train_c):.4f}")
print(f"   测试集准确率: {accuracy_cancer:.4f}")
print(f"   ROC-AUC 分数: {auc_score:.4f}")

# 12. 绘制可视化图表
print("\n12. 生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 12.1 不同深度的性能对比
ax = axes[0, 0]
ax.plot(depths, train_scores, 'o-', label='训练集', linewidth=2, markersize=8)
ax.plot(depths, test_scores, 's-', label='测试集', linewidth=2, markersize=8)
ax.set_xlabel('弱学习器深度', fontsize=11)
ax.set_ylabel('准确率', fontsize=11)
ax.set_title('不同弱学习器深度的性能', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(depths)

# 12.2 弱学习器权重分析
ax = axes[0, 1]
n_show = min(20, len(estimator_weights))
ax.bar(range(n_show), estimator_weights[:n_show], alpha=0.7, color='steelblue')
ax.set_xlabel('弱学习器索引', fontsize=11)
ax.set_ylabel('权重', fontsize=11)
ax.set_title('前20个弱学习器的权重', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 12.3 混淆矩阵热力图
ax = axes[1, 0]
cm_display = cm_ada.astype('float') / cm_ada.sum(axis=1)[:, np.newaxis]
im = ax.imshow(cm_display, cmap='Blues', aspect='auto')
ax.set_xlabel('预测标签', fontsize=11)
ax.set_ylabel('真实标签', fontsize=11)
ax.set_title('混淆矩阵（归一化）', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(target_names)))
ax.set_yticks(range(len(target_names)))
ax.set_xticklabels(target_names_display)
ax.set_yticklabels(target_names_display)

# 添加数值标签
for i in range(len(target_names)):
    for j in range(len(target_names)):
        text = ax.text(j, i, f'{cm_display[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=ax)

# 12.4 模型对比
ax = axes[1, 1]
models = ['单决策树\n(深度=1)', 'AdaBoost\n基础模型', 'AdaBoost\n优化模型']
train_accs = [
    dt_single.score(X_train, y_train),
    ada_basic.score(X_train, y_train),
    ada_best.score(X_train, y_train)
]
test_accs = [
    dt_single.score(X_test, y_test),
    ada_basic.score(X_test, y_test),
    ada_best.score(X_test, y_test)
]

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, train_accs, width, label='训练集', alpha=0.8, color='skyblue')
ax.bar(x + width/2, test_accs, width, label='测试集', alpha=0.8, color='orange')

ax.set_ylabel('准确率', fontsize=11)
ax.set_title('模型性能对比', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# 添加数值标签
for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
    ax.text(i - width/2, train_acc + 0.02, f'{train_acc:.3f}', 
            ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, test_acc + 0.02, f'{test_acc:.3f}', 
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('adaboost_analysis.png', dpi=150, bbox_inches='tight')
print("   图表已保存为 'adaboost_analysis.png'")

# 绘制第二个图表：乳腺癌数据集的 ROC 曲线
fig2, ax2 = plt.subplots(figsize=(10, 8))

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test_c, y_pred_proba_cancer)

# 绘制 ROC 曲线
ax2.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC 曲线 (AUC = {auc_score:.4f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('假正率 (FPR)', fontsize=11)
ax2.set_ylabel('真正率 (TPR)', fontsize=11)
ax2.set_title('AdaBoost - 乳腺癌数据集 ROC 曲线', fontsize=12, fontweight='bold')
ax2.legend(loc="lower right", fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('adaboost_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("   ROC 曲线已保存为 'adaboost_roc_curve.png'")

print("\n" + "=" * 70)
print("AdaBoost Demo 完成！")
print("=" * 70)

# 13. 总结
print("\n13. 总结与说明:")
print("   - AdaBoost 是一种集成学习算法，通过加权多个弱学习器来提高性能")
print("   - 通常使用浅决策树（如深度=1 的树）作为弱学习器")
print("   - 关键参数：")
print("     * n_estimators: 弱学习器的个数")
print("     * learning_rate: 学习率，控制每个弱学习器的权重影响")
print("     * algorithm: 'SAMME'（支持多分类）或 'SAMME.R'（概率输出）")
print("   - 优点：能有效减少偏差和方差、处理不平衡数据、可解释性好")
print("   - 缺点：对噪声敏感、训练速度相对较慢（不能并行化）")

