import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体，解决 matplotlib 无法显示中文的问题
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def multinomial_logistic_regression_example():
    """
    逻辑回归（Logistic Regression）多分类完整示例
    使用经典鸢尾花（Iris）数据集进行多分类：判断鸢尾花的种类。
    """
    print("=== 1. 加载数据 ===")
    # 加载鸢尾花数据集
    iris_data = load_iris()
    X = iris_data.data      # 特征矩阵 (150个样本, 4个特征: 萼片长度、萼片宽度、花瓣长度、花瓣宽度)
    y = iris_data.target    # 目标标签 (0: setosa, 1: versicolor, 2: virginica)
    
    # 将英文标签替换为中文标签
    target_names_zh = ['山鸢尾 (Setosa)', '变色鸢尾 (Versicolor)', '维吉尼亚鸢尾 (Virginica)']
    
    print(f"特征数据的形状: {X.shape}")
    print(f"目标标签的形状: {y.shape}")
    print(f"类别名称: {target_names_zh}\n")

    print("=== 2. 数据集划分 ===")
    # 将数据集划分为训练集和测试集（测试集占25%）
    # stratify=y 确保训练集和测试集中的三类鸢尾花比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}\n")

    print("=== 3. 特征标准化 ===")
    # 逻辑回归对特征尺度敏感，需进行标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("标准化处理完成。\n")

    print("=== 4. 构建并训练模型 ===")
    # 实例化逻辑回归模型
    # solver='lbfgs': 优化算法，默认支持多分类
    # max_iter=1000: 增加最大迭代次数，保证模型收敛
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    # 在训练集上训练模型
    model.fit(X_train_scaled, y_train)
    print("模型训练完成。\n")

    print("=== 5. 模型预测与评估 ===")
    # 对测试集进行预测
    y_pred = model.predict(X_test_scaled)
    
    # 5.1 准确率 (Accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型在测试集上的准确率: {accuracy:.4f}\n")

    # 5.2 分类报告 (Classification Report)
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names_zh))

    # 5.3 获取预测概率
    y_pred_proba = model.predict_proba(X_test_scaled)
    print("前5个测试样本的预测概率: ")
    print(np.round(y_pred_proba[:5], 3))
    print("前5个测试样本的预测类别 (0: 山鸢尾, 1: 变色鸢尾, 2: 维吉尼亚鸢尾):")
    print(y_pred[:5], "\n")

    print("=== 6. 可视化混淆矩阵 ===")
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 绘制热力图展示混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=target_names_zh, 
                yticklabels=target_names_zh)
    plt.title('鸢尾花多分类混淆矩阵 (Confusion Matrix)')
    plt.xlabel('预测类别 (Predicted Label)')
    plt.ylabel('真实类别 (True Label)')
    
    # 为了保证类别标签不被遮挡，可以稍微旋转一下 x 轴标签
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 运行逻辑回归多分类完整示例
    multinomial_logistic_regression_example()
