import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def logistic_regression_example():
    """
    逻辑回归（Logistic Regression）完整示例
    使用经典乳腺癌（Breast Cancer）数据集进行二分类：判断肿瘤是良性还是恶性。
    """
    print("=== 1. 加载数据 ===")
    # 加载乳腺癌数据集
    cancer_data = load_breast_cancer()
    X = cancer_data.data      # 特征矩阵 (569个样本, 30个特征)
    y = cancer_data.target    # 目标标签 (0: 恶性, 1: 良性)
    
    target_names_zh = ['恶性', '良性']
    print(f"特征数据的形状: {X.shape}")
    print(f"目标标签的形状: {y.shape}")
    print(f"类别名称: {target_names_zh}\n")

    print("=== 2. 数据集划分 ===")
    # 将数据集划分为训练集和测试集（测试集占25%，相当于 8:2 或者是 7.5:2.5 左右）
    # random_state=42 保证每次运行的结果一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}\n")

    print("=== 3. 特征标准化 ===")
    # 逻辑回归对特征的尺度非常敏感，通常需要进行标准化 (均值为0，方差为1)
    scaler = StandardScaler()
    # 使用训练集计算均值和标准差并进行变换
    X_train_scaled = scaler.fit_transform(X_train)
    # 使用刚才训练集上的标准去变换测试集 (防止信息泄露)
    X_test_scaled = scaler.transform(X_test)
    print("标准化处理完成。\n")

    print("=== 4. 构建并训练模型 ===")
    # 实例化逻辑回归模型
    # max_iter 指定最大迭代次数，如果遇到收敛警告说明需要增大迭代次数或进行数据预处理
    model = LogisticRegression(max_iter=1000)
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
    # 包含精确率(Precision)、召回率(Recall)、F1分数等指标
    print("分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names_zh))

    # 5.3 获取预测概率 (可选)
    # 逻辑回归不仅可以输出分类结果，还可以给出每种类的概率
    y_pred_proba = model.predict_proba(X_test_scaled)
    print("前5个测试样本的预测概率: ")
    print(np.round(y_pred_proba[:5], 3))
    print("前5个测试样本的预测类别:\n", y_pred[:5], "\n")

    print("=== 6. 可视化混淆矩阵 ===")
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 绘制热力图展示混淆矩阵
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names_zh, 
                yticklabels=target_names_zh)
    plt.title('混淆矩阵 (Confusion Matrix)')
    plt.xlabel('预测类别 (Predicted Label)')
    plt.ylabel('真实类别 (True Label)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 运行逻辑回归完整示例
    logistic_regression_example()
