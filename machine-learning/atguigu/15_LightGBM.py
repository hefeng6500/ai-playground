"""
LightGBM (Light Gradient Boosting Machine) Demo
包含分类、回归和特征重要性分析等示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)
import warnings
import sys
import io

try:
    import lightgbm as lgb
except ImportError as e:
    print("LightGBM 未安装，请运行: pip install lightgbm")
    print(f"错误详情：{e}")
    sys.exit(1)

# 设置 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang HK', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True
plt.rcParams['mathtext.default'] = 'regular'

# 特征名称中文翻译映射表
FEATURE_NAME_ZH = {
    'mean radius': '平均半径',
    'mean texture': '平均纹理',
    'mean perimeter': '平均周长',
    'mean area': '平均面积',
    'mean smoothness': '平均光滑度',
    'mean compactness': '平均紧凑度',
    'mean concavity': '平均凹陷度',
    'mean concave points': '平均凹陷点数',
    'mean symmetry': '平均对称性',
    'mean fractal dimension': '平均分形维数',
    'radius error': '半径误差',
    'texture error': '纹理误差',
    'perimeter error': '周长误差',
    'area error': '面积误差',
    'smoothness error': '光滑度误差',
    'compactness error': '紧凑度误差',
    'concavity error': '凹陷度误差',
    'concave points error': '凹陷点数误差',
    'symmetry error': '对称性误差',
    'fractal dimension error': '分形维数误差',
    'worst radius': '最差半径',
    'worst texture': '最差纹理',
    'worst perimeter': '最差周长',
    'worst area': '最差面积',
    'worst smoothness': '最差光滑度',
    'worst compactness': '最差紧凑度',
    'worst concavity': '最差凹陷度',
    'worst concave points': '最差凹陷点数',
    'worst symmetry': '最差对称性',
    'worst fractal dimension': '最差分形维数',
}


def demo_classification():
    """LightGBM 分类示例 - 使用乳腺癌数据集"""
    print("=" * 60)
    print("LightGBM - 分类示例 (乳腺癌数据集)")
    print("=" * 60)
    
    # 加载数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n数据集信息：")
    print(f"  训练集样本数：{X_train.shape[0]}")
    print(f"  测试集样本数：{X_test.shape[0]}")
    print(f"  特征数：{X_train.shape[1]}")
    print(f"  类别：{np.unique(y)}")
    
    # 创建并训练 LightGBM 分类器
    print(f"\n训练 LightGBM 分类器...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    clf.fit(X_train, y_train)
    
    # 预测
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # 评估
    print(f"\n分类性能指标：")
    print(f"  准确率 (Accuracy)：{accuracy_score(y_test, y_pred):.4f}")
    print(f"  精确率 (Precision)：{precision_score(y_test, y_pred):.4f}")
    print(f"  召回率 (Recall)：{recall_score(y_test, y_pred):.4f}")
    print(f"  F1 分数：{f1_score(y_test, y_pred):.4f}")
    
    # 交叉验证
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n5 折交叉验证准确率：{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 特征重要性
    feature_importance = clf.feature_importances_
    feature_names = data.feature_names
    
    print(f"\n前 10 个最重要的特征：")
    top_indices = np.argsort(feature_importance)[::-1][:10]
    for idx, importance in enumerate(feature_importance[top_indices], 1):
        feature_name_en = feature_names[top_indices[idx-1]]
        feature_name_zh = FEATURE_NAME_ZH.get(feature_name_en, feature_name_en)
        print(f"  {idx}. {feature_name_zh}：{importance:.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 混淆矩阵
    axes[0].imshow(cm, cmap='Blues', aspect='auto')
    axes[0].set_title('混淆矩阵', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i][j], ha='center', va='center', color='red', fontsize=14)
    
    # 特征重要性（前 10）
    top_indices = np.argsort(feature_importance)[::-1][:10]
    top_features = [FEATURE_NAME_ZH.get(feature_names[i], feature_names[i]) for i in top_indices]
    top_importance = feature_importance[top_indices]
    
    axes[1].barh(range(len(top_features)), top_importance, color='steelblue')
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features, fontsize=9)
    axes[1].set_xlabel('特征重要性', fontsize=11)
    axes[1].set_title('前 10 个最重要的特征', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def demo_regression():
    """LightGBM 回归示例"""
    print("\n" + "=" * 60)
    print("LightGBM - 回归示例")
    print("=" * 60)
    
    # 生成回归数据集
    X, y = make_regression(
        n_samples=300,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据集信息：")
    print(f"  训练集样本数：{X_train.shape[0]}")
    print(f"  测试集样本数：{X_test.shape[0]}")
    print(f"  特征数：{X_train.shape[1]}")
    
    # 创建并训练 LightGBM 回归器
    print(f"\n训练 LightGBM 回归器...")
    reg = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    reg.fit(X_train, y_train)
    
    # 预测
    y_pred = reg.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n回归性能指标：")
    print(f"  均方误差 (MSE)：{mse:.4f}")
    print(f"  均方根误差 (RMSE)：{rmse:.4f}")
    print(f"  平均绝对误差 (MAE)：{mae:.4f}")
    print(f"  决定系数 (R²)：{r2:.4f}")
    
    # 交叉验证
    cv_scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='r2')
    print(f"\n5 折交叉验证 R² 分数：{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 绘制预测效果
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 预测值 vs 真实值
    axes[0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='k', linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
    axes[0].set_xlabel('真实值', fontsize=11)
    axes[0].set_ylabel('预测值', fontsize=11)
    axes[0].set_title('预测值 vs 真实值', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 残差分析
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='steelblue', edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('预测值', fontsize=11)
    axes[1].set_ylabel('残差', fontsize=11)
    axes[1].set_title('残差分析', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demo_learning_curve():
    """LightGBM 学习曲线示例"""
    print("\n" + "=" * 60)
    print("LightGBM - 学习曲线分析")
    print("=" * 60)
    
    # 生成分类数据
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建不同参数的模型并记录训练过程
    print(f"\n分析不同 max_depth 值的影响...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for max_depth in [3, 5, 7, 10]:
        clf = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        train_scores = []
        test_scores = []
        
        for n_est in range(5, 51, 5):
            clf.set_params(n_estimators=n_est)
            clf.fit(X_train, y_train)
            train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
            test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
        
        ax.plot(range(5, 51, 5), test_scores, marker='o', label=f'max_depth={max_depth}')
    
    ax.set_xlabel('树的数量 (n_estimators)', fontsize=11)
    ax.set_ylabel('准确率', fontsize=11)
    ax.set_title('LightGBM 学习曲线 - 不同树深的影响', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demo_hyperparameter_tuning():
    """LightGBM 超参数调优示例"""
    print("\n" + "=" * 60)
    print("LightGBM - 超参数调优示例")
    print("=" * 60)
    
    # 生成数据
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n网格搜索最优超参数...")
    
    best_score = 0
    best_params = {}
    results = []
    
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    num_leaves = [15, 31, 50, 100]
    
    for lr in learning_rates:
        for leaves in num_leaves:
            clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=lr,
                num_leaves=leaves,
                random_state=42,
                verbose=-1
            )
            
            clf.fit(X_train, y_train)
            score = accuracy_score(y_test, clf.predict(X_test))
            
            results.append({
                'learning_rate': lr,
                'num_leaves': leaves,
                'accuracy': score
            })
            
            if score > best_score:
                best_score = score
                best_params = {'learning_rate': lr, 'num_leaves': leaves}
    
    print(f"\n最佳超参数组合：")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  num_leaves: {best_params['num_leaves']}")
    print(f"  最佳准确率: {best_score:.4f}")
    
    # 绘制超参数影响热力图
    results_df = pd.DataFrame(results)
    pivot_table = results_df.pivot_table(
        values='accuracy',
        index='learning_rate',
        columns='num_leaves'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_table, cmap='RdYlGn', aspect='auto', vmin=0.85, vmax=1.0)
    
    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    ax.set_xlabel('num_leaves', fontsize=11)
    ax.set_ylabel('learning_rate', fontsize=11)
    ax.set_title('LightGBM 超参数调优热力图', fontsize=12, fontweight='bold')
    
    # 添加数值标签
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            text = ax.text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    fig.colorbar(im, ax=ax, label='准确率')
    plt.tight_layout()
    plt.show()


def demo_comparison():
    """LightGBM 与其他模型的性能对比"""
    print("\n" + "=" * 60)
    print("LightGBM - 与其他模型的性能对比")
    print("=" * 60)
    
    # 生成数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n训练不同模型...")
    
    models = {}
    
    # LightGBM
    lgb_clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    lgb_clf.fit(X_train, y_train)
    models['LightGBM'] = accuracy_score(y_test, lgb_clf.predict(X_test))
    
    # 尝试导入其他模型进行对比
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        # RandomForest
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        models['RandomForest'] = accuracy_score(y_test, rf_clf.predict(X_test))
        
        # GradientBoosting
        gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_clf.fit(X_train, y_train)
        models['GradientBoosting'] = accuracy_score(y_test, gb_clf.predict(X_test))
        
    except ImportError:
        pass
    
    try:
        import xgboost as xgb
        
        # XGBoost
        xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        xgb_clf.fit(X_train, y_train)
        models['XGBoost'] = accuracy_score(y_test, xgb_clf.predict(X_test))
        
    except ImportError:
        pass
    
    # 绘制对比结果
    print(f"\n模型性能对比：")
    for model_name, accuracy in models.items():
        print(f"  {model_name}：{accuracy:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(models.keys())
    accuracies = list(models.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(model_names)]
    
    bars = ax.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('不同模型性能对比', fontsize=13, fontweight='bold')
    ax.set_ylim([0.8, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("LightGBM (Light Gradient Boosting Machine) 完整示例")
    print("=" * 60)
    print(f"LightGBM 版本：{lgb.__version__}")
    
    # 运行所有演示
    demo_classification()
    demo_regression()
    demo_learning_curve()
    demo_hyperparameter_tuning()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("✓ 所有示例执行完毕！")
    print("=" * 60)


if __name__ == '__main__':
    main()

