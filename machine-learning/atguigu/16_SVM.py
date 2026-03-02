"""
支持向量机(Support Vector Machine, SVM) Demo
演示SVM在分类和回归中的应用，包括不同核函数的对比
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 线性SVM分类 ====================
print("=" * 50)
print("1. 线性SVM分类演示")
print("=" * 50)

# 生成线性可分数据
X_linear, y_linear = make_classification(n_samples=100, n_features=2, n_classes=2,
                                         n_clusters_per_class=1, n_redundant=0,
                                         random_state=42)

# 使用线性核的SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_linear, y_linear)

# 预测和评估
y_pred_linear = svm_linear.predict(X_linear)
print(f"线性SVM准确率: {accuracy_score(y_linear, y_pred_linear):.4f}")
print(f"支持向量个数: {len(svm_linear.support_vectors_)}")

# 绘制决策边界
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1.1 线性SVM
ax = axes[0, 0]
h = 0.02
x_min, x_max = X_linear[:, 0].min() - 0.5, X_linear[:, 0].max() + 0.5
y_min, y_max = X_linear[:, 1].min() - 0.5, X_linear[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=20, cmap='RdBu')
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
ax.scatter(X_linear[y_linear == 0, 0], X_linear[y_linear == 0, 1],
          c='red', marker='o', s=50, label='Class 0')
ax.scatter(X_linear[y_linear == 1, 0], X_linear[y_linear == 1, 1],
          c='blue', marker='s', s=50, label='Class 1')
ax.scatter(svm_linear.support_vectors_[:, 0], svm_linear.support_vectors_[:, 1],
          s=200, linewidth=1.5, facecolors='none', edgecolors='green', label='Support Vectors')
ax.set_title('线性SVM (Linear Kernel)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()

# ==================== 2. 非线性SVM - RBF核 ====================
print("\n" + "=" * 50)
print("2. 非线性SVM (RBF核) 演示")
print("=" * 50)

# 生成非线性数据
X_nonlinear, y_nonlinear = make_circles(n_samples=100, noise=0.1, random_state=42)
X_nonlinear = StandardScaler().fit_transform(X_nonlinear)

# RBF核SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_nonlinear, y_nonlinear)

y_pred_rbf = svm_rbf.predict(X_nonlinear)
print(f"RBF SVM准确率: {accuracy_score(y_nonlinear, y_pred_rbf):.4f}")
print(f"支持向量个数: {len(svm_rbf.support_vectors_)}")

# 2.1 绘制RBF SVM决策边界
ax = axes[0, 1]
x_min, x_max = X_nonlinear[:, 0].min() - 0.5, X_nonlinear[:, 0].max() + 0.5
y_min, y_max = X_nonlinear[:, 1].min() - 0.5, X_nonlinear[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=20, cmap='RdBu')
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
ax.scatter(X_nonlinear[y_nonlinear == 0, 0], X_nonlinear[y_nonlinear == 0, 1],
          c='red', marker='o', s=50, label='Class 0')
ax.scatter(X_nonlinear[y_nonlinear == 1, 0], X_nonlinear[y_nonlinear == 1, 1],
          c='blue', marker='s', s=50, label='Class 1')
ax.scatter(svm_rbf.support_vectors_[:, 0], svm_rbf.support_vectors_[:, 1],
          s=200, linewidth=1.5, facecolors='none', edgecolors='green', label='Support Vectors')
ax.set_title('非线性SVM (RBF Kernel)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()

# ==================== 3. 不同C参数对SVM的影响 ====================
print("\n" + "=" * 50)
print("3. 正则化参数C的影响")
print("=" * 50)

# 生成有噪声的数据
X_noise, y_noise = make_classification(n_samples=100, n_features=2, n_classes=2,
                                       n_clusters_per_class=1, n_redundant=0,
                                       random_state=42)

# 3.1 C较小（更多正则化）
svm_c_small = SVC(kernel='rbf', C=0.1, gamma='scale', random_state=42)
svm_c_small.fit(X_noise, y_noise)
y_pred_c_small = svm_c_small.predict(X_noise)

print(f"C=0.1时准确率: {accuracy_score(y_noise, y_pred_c_small):.4f}, 支持向量数: {len(svm_c_small.support_vectors_)}")

ax = axes[1, 0]
x_min, x_max = X_noise[:, 0].min() - 0.5, X_noise[:, 0].max() + 0.5
y_min, y_max = X_noise[:, 1].min() - 0.5, X_noise[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_c_small.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=20, cmap='RdBu')
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
ax.scatter(X_noise[y_noise == 0, 0], X_noise[y_noise == 0, 1],
          c='red', marker='o', s=50, label='Class 0')
ax.scatter(X_noise[y_noise == 1, 0], X_noise[y_noise == 1, 1],
          c='blue', marker='s', s=50, label='Class 1')
ax.scatter(svm_c_small.support_vectors_[:, 0], svm_c_small.support_vectors_[:, 1],
          s=200, linewidth=1.5, facecolors='none', edgecolors='green', label='Support Vectors')
ax.set_title('C=0.1 (更多正则化，欠拟合)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()

# 3.2 C较大（较少正则化）
svm_c_large = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
svm_c_large.fit(X_noise, y_noise)
y_pred_c_large = svm_c_large.predict(X_noise)

print(f"C=100时准确率: {accuracy_score(y_noise, y_pred_c_large):.4f}, 支持向量数: {len(svm_c_large.support_vectors_)}")

ax = axes[1, 1]
x_min, x_max = X_noise[:, 0].min() - 0.5, X_noise[:, 0].max() + 0.5
y_min, y_max = X_noise[:, 1].min() - 0.5, X_noise[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_c_large.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=20, cmap='RdBu')
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
ax.scatter(X_noise[y_noise == 0, 0], X_noise[y_noise == 0, 1],
          c='red', marker='o', s=50, label='Class 0')
ax.scatter(X_noise[y_noise == 1, 0], X_noise[y_noise == 1, 1],
          c='blue', marker='s', s=50, label='Class 1')
ax.scatter(svm_c_large.support_vectors_[:, 0], svm_c_large.support_vectors_[:, 1],
          s=200, linewidth=1.5, facecolors='none', edgecolors='green', label='Support Vectors')
ax.set_title('C=100 (较少正则化，过拟合)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()

plt.tight_layout()
plt.show()

# ==================== 4. SVR回归演示 ====================
print("\n" + "=" * 50)
print("4. 支持向量回归 (SVR) 演示")
print("=" * 50)

# 生成回归数据
np.random.seed(42)
X_reg = np.linspace(0, 10, 100).reshape(-1, 1)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, 100)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 创建和训练SVR模型
svr_rbf = SVR(kernel='rbf', C=100, gamma='auto')
svr_rbf.fit(X_train, y_train)

# 预测
y_pred_train = svr_rbf.predict(X_train)
y_pred_test = svr_rbf.predict(X_test)

print(f"SVR训练集R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"SVR测试集R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"SVR测试集MSE: {mean_squared_error(y_test, y_pred_test):.4f}")

# 绘制回归结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4.1 SVR预测结果
ax = axes[0]
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
y_plot = svr_rbf.predict(X_plot)

ax.scatter(X_train, y_train, c='red', s=50, label='Training Data', alpha=0.6)
ax.scatter(X_test, y_test, c='blue', s=50, label='Test Data', alpha=0.6)
ax.plot(X_plot, y_plot, 'g-', linewidth=2, label='SVR Prediction')
ax.plot(X_plot, np.sin(X_plot), 'k--', linewidth=1, label='True Function', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('支持向量回归 (SVR with RBF Kernel)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4.2 不同核函数的对比
ax = axes[1]
kernels = ['linear', 'rbf', 'poly']
colors = ['red', 'blue', 'green']

for kernel, color in zip(kernels, colors):
    svr = SVR(kernel=kernel, C=100, gamma='auto', degree=3)
    svr.fit(X_train, y_train)
    y_plot = svr.predict(X_plot)
    r2 = r2_score(y_test, svr.predict(X_test))
    ax.plot(X_plot, y_plot, color=color, linewidth=2, label=f'{kernel.upper()} (R²={r2:.3f})')

ax.plot(X_plot, np.sin(X_plot), 'k--', linewidth=1, label='True Function', alpha=0.5)
ax.scatter(X_train, y_train, c='gray', s=30, alpha=0.3)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('不同核函数的SVR对比')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 5. 多分类SVM ====================
print("\n" + "=" * 50)
print("5. 多分类SVM演示")
print("=" * 50)

# 生成多分类数据
X_multi, y_multi = make_classification(n_samples=300, n_features=2, n_classes=3,
                                       n_clusters_per_class=1, n_redundant=0,
                                       n_informative=2, random_state=42)

# 训练多分类SVM
svm_multi = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_multi.fit(X_multi, y_multi)

y_pred_multi = svm_multi.predict(X_multi)
accuracy_multi = accuracy_score(y_multi, y_pred_multi)

print(f"多分类SVM准确率: {accuracy_multi:.4f}")
print("\n分类报告:")
print(classification_report(y_multi, y_pred_multi))

# 绘制多分类结果
fig, ax = plt.subplots(figsize=(10, 8))

h = 0.02
x_min, x_max = X_multi[:, 0].min() - 0.5, X_multi[:, 0].max() + 0.5
y_min, y_max = X_multi[:, 1].min() - 0.5, X_multi[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_multi.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
colors = ['red', 'green', 'blue']
for i in range(3):
    ax.scatter(X_multi[y_multi == i, 0], X_multi[y_multi == i, 1],
              c=colors[i], marker='o', s=50, label=f'Class {i}', edgecolors='black')

ax.set_title('多分类SVM (3-class Classification)')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 6. 总结 ====================
print("\n" + "=" * 50)
print("6. SVM总结")
print("=" * 50)
print("""
SVM的主要特点：
1. 核函数(Kernel)：
   - linear: 线性核，用于线性可分数据
   - rbf: 高斯核，用于非线性数据
   - poly: 多项式核
   - sigmoid: sigmoid核

2. 正则化参数C：
   - C越小，容错能力越强，可能欠拟合
   - C越大，容错能力越弱，可能过拟合

3. gamma参数：
   - 只在rbf、poly、sigmoid核中有效
   - gamma越大，决策边界越复杂
   - gamma越小，决策边界越平滑

4. 支持向量：
   - 只有部分样本对决策边界有贡献
   - 支持向量少意味着泛化能力强

5. 应用场景：
   - 二分类和多分类问题
   - 高维数据分类
   - 回归问题(SVR)
   - 异常检测
""")


