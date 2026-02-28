"""
梯度提升决策树 (GBDT - Gradient Boosting Decision Tree) Demo
包含分类和回归两个示例
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
import sys
import io

# 设置 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang HK', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True
# 启用 LaTeX 渲染以支持上标
plt.rcParams['text.usetex'] = False  # 关闭完整 LaTeX 以避免依赖问题
plt.rcParams['mathtext.default'] = 'regular'  # 使用数学文本


def demo_classification():
    """梯度提升分类示例"""
    print("=" * 60)
    print("梯度提升决策树 - 分类示例")
    print("=" * 60)
    
    # 生成样本数据
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42,
        class_sep=1.0
    )
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据集信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征数: {X_train.shape[1]}")
    
    # 创建和训练梯度提升分类器
    print(f"\n训练梯度提升分类器...")
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,        # 树的数量
        learning_rate=0.1,       # 学习率
        max_depth=5,             # 树的最大深度
        min_samples_split=5,     # 分裂所需的最小样本数
        min_samples_leaf=2,      # 叶子节点所需的最小样本数
        subsample=0.8,           # 样本采样比例
        random_state=42
    )
    gb_clf.fit(X_train, y_train)
    
    # 预测
    y_train_pred = gb_clf.predict(X_train)
    y_test_pred = gb_clf.predict(X_test)
    
    # 评估
    print(f"\n分类性能评估:")
    print(f"  训练集准确率: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  测试集准确率: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  精确率 (Precision): {precision_score(y_test, y_test_pred):.4f}")
    print(f"  召回率 (Recall): {recall_score(y_test, y_test_pred):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_test_pred):.4f}")
    
    # 特征重要性
    print(f"\n前5个重要特征:")
    feature_importance = gb_clf.feature_importances_
    top_indices = np.argsort(feature_importance)[-5:][::-1]
    for idx in top_indices:
        print(f"  特征 {idx}: {feature_importance[idx]:.4f}")
    
    # 绘制特征重要性
    plt.figure(figsize=(10, 6))
    indices = np.argsort(feature_importance)
    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), [f'特征 {i}' for i in indices])
    plt.xlabel('重要性')
    plt.title('梯度提升分类器 - 特征重要性')
    plt.tight_layout()
    # plt.savefig('gb_classification_importance.png', dpi=100)
    plt.show()
    
    # 绘制训练过程中的性能变化
    train_scores = []
    test_scores = []
    for i, pred_train in enumerate(gb_clf.staged_predict(X_train)):
        train_scores.append(accuracy_score(y_train, pred_train))
    for i, pred_test in enumerate(gb_clf.staged_predict(X_test)):
        test_scores.append(accuracy_score(y_test, pred_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='训练集', linewidth=2)
    plt.plot(test_scores, label='测试集', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.title('梯度提升分类器 - 训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    # plt.savefig('gb_classification_training.png', dpi=100)
    plt.show()


def demo_regression():
    """梯度提升回归示例"""
    print("\n" + "=" * 60)
    print("梯度提升决策树 - 回归示例")
    print("=" * 60)
    
    # 生成样本数据
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        noise=10.0,
        random_state=42
    )
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据集信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征数: {X_train.shape[1]}")
    
    # 创建和训练梯度提升回归器
    print(f"\n训练梯度提升回归器...")
    gb_reg = GradientBoostingRegressor(
        n_estimators=100,        # 树的数量
        learning_rate=0.1,       # 学习率
        max_depth=5,             # 树的最大深度
        min_samples_split=5,     # 分裂所需的最小样本数
        min_samples_leaf=2,      # 叶子节点所需的最小样本数
        subsample=0.8,           # 样本采样比例
        random_state=42
    )
    gb_reg.fit(X_train, y_train)
    
    # 预测
    y_train_pred = gb_reg.predict(X_train)
    y_test_pred = gb_reg.predict(X_test)
    
    # 评估
    print(f"\n回归性能评估:")
    print(f"  训练集 R²: {r2_score(y_train, y_train_pred):.4f}")
    print(f"  测试集 R²: {r2_score(y_test, y_test_pred):.4f}")
    print(f"  测试集 MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f"  测试集 MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"  测试集 RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
    
    # 特征重要性
    print(f"\n前5个重要特征:")
    feature_importance = gb_reg.feature_importances_
    top_indices = np.argsort(feature_importance)[-5:][::-1]
    for idx in top_indices:
        print(f"  特征 {idx}: {feature_importance[idx]:.4f}")
    
    # 绘制特征重要性
    plt.figure(figsize=(10, 6))
    indices = np.argsort(feature_importance)
    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), [f'特征 {i}' for i in indices])
    plt.xlabel('重要性')
    plt.title('梯度提升回归器 - 特征重要性')
    plt.tight_layout()
    # plt.savefig('gb_regression_importance.png', dpi=100)
    plt.show()
    
    # 绘制预测vs实际
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5, label='训练集')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('训练集 - 预测 vs 实际')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5, label='测试集')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('测试集 - 预测 vs 实际')
    plt.legend()
    
    plt.tight_layout()
    # plt.savefig('gb_regression_prediction.png', dpi=100)
    plt.show()
    
    # 绘制训练过程中的性能变化
    train_scores = []
    test_scores = []
    for pred_train in gb_reg.staged_predict(X_train):
        train_scores.append(r2_score(y_train, pred_train))
    for pred_test in gb_reg.staged_predict(X_test):
        test_scores.append(r2_score(y_test, pred_test))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='训练集', linewidth=2)
    plt.plot(test_scores, label='测试集', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel(r'$R^2$ Score')  # 使用 LaTeX 格式显示上标
    plt.title('梯度提升回归器 - 训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    # plt.savefig('gb_regression_training.png', dpi=100)
    plt.show()


def hyperparameter_tuning_demo():
    """超参数调优示例"""
    print("\n" + "=" * 60)
    print("梯度提升决策树 - 超参数调优示例")
    print("=" * 60)
    
    # 生成样本数据
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42,
        class_sep=1.0
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 测试不同的学习率
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    train_scores = []
    test_scores = []
    
    print(f"\n测试不同的学习率:")
    for lr in learning_rates:
        gb_clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=lr,
            max_depth=5,
            random_state=42
        )
        gb_clf.fit(X_train, y_train)
        train_score = gb_clf.score(X_train, y_train)
        test_score = gb_clf.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"  学习率={lr}: 训练准确率={train_score:.4f}, 测试准确率={test_score:.4f}")
    
    # 绘制学习率影响
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, train_scores, 'o-', label='训练集', linewidth=2, markersize=8)
    plt.plot(learning_rates, test_scores, 's-', label='测试集', linewidth=2, markersize=8)
    plt.xlabel('学习率')
    plt.ylabel('准确率')
    plt.title('学习率对模型性能的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    # plt.savefig('gb_learning_rate_impact.png', dpi=100)
    plt.show()
    
    # 测试不同的树深度
    max_depths = [2, 3, 4, 5, 6, 7, 8]
    train_scores = []
    test_scores = []
    
    print(f"\n测试不同的树最大深度:")
    for depth in max_depths:
        gb_clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=depth,
            random_state=42
        )
        gb_clf.fit(X_train, y_train)
        train_score = gb_clf.score(X_train, y_train)
        test_score = gb_clf.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"  树深度={depth}: 训练准确率={train_score:.4f}, 测试准确率={test_score:.4f}")
    
    # 绘制树深度影响
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_scores, 'o-', label='训练集', linewidth=2, markersize=8)
    plt.plot(max_depths, test_scores, 's-', label='测试集', linewidth=2, markersize=8)
    plt.xlabel('树最大深度')
    plt.ylabel('准确率')
    plt.title('树深度对模型性能的影响')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    # plt.savefig('gb_max_depth_impact.png', dpi=100)
    plt.show()


def gbdt_explanation():
    """GBDT 算法解释"""
    print("\n" + "=" * 60)
    print("梯度提升决策树 (GBDT) 算法原理")
    print("=" * 60)
    
    explanation = """
【GBDT 基本原理】
1. 初始化：用常数模型初始化 (如平均值)
2. 迭代过程：
   - 计算负梯度 (伪残差)
   - 拟合一个决策树到伪残差
   - 计算最优步长
   - 更新预测值
3. 最终预测：所有树的预测加权求和

【关键参数说明】
- n_estimators: 弱学习器 (决策树) 的数量，通常在 50-500 之间
- learning_rate: 学习率，控制每棵树的贡献度 (0-1)，越小过拟合风险越低
- max_depth: 树的最大深度，通常在 3-10 之间
- min_samples_split: 分裂内部节点所需的最小样本数
- min_samples_leaf: 叶子节点所需的最小样本数
- subsample: 用于拟合每个弱学习器的样本比例，< 1.0 可以减少过拟合

【GBDT 优点】
[+] 预测性能好，常在竞赛中获胜
[+] 能处理混合类型特征
[+] 可以处理非线性关系
[+] 提供特征重要性评估
[+] 支持分类和回归任务

【GBDT 缺点】
[-] 计算复杂，训练时间长
[-] 超参数较多，调优困难
[-] 对异常值敏感
[-] 模型不易解释

【应用场景】
- 点击率预测 (CTR)
- 排序问题 (Learning to Rank)
- 回归预测
- 分类问题
- 特征工程
"""
    
    print(explanation)


if __name__ == "__main__":
    # 运行所有示例
    demo_classification()
    demo_regression()
    hyperparameter_tuning_demo()
    gbdt_explanation()
    
    print("\n" + "=" * 60)
    print("所有演示已完成！")
    print("=" * 60)

