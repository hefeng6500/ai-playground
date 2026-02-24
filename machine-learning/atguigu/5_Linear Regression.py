import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 完整的线性回归 —— 梯度下降法实现 + 可视化
# 目标函数: y = 4 + 3x + 噪声
# ============================================================

# ---------- 1. 准备数据 ----------
np.random.seed(42)
m = 100  # 样本数量
X = 2 * np.random.rand(m, 1)                # X ∈ [0, 2)
y = 4 + 3 * X + np.random.randn(m, 1)       # y = 4 + 3x + ε

# 添加偏置列 x0=1，使 X_b = [1, x]
X_b = np.c_[np.ones((m, 1)), X]

# ---------- 2. 初始化超参数 ----------
theta = np.random.randn(2, 1)   # 随机初始化 [theta_0, theta_1]
learning_rate = 0.1
n_iterations = 1000

# ---------- 3. 梯度下降 ----------
cost_history = []                # 记录每轮的损失值（MSE）
theta_history = [theta.copy()]   # 记录参数变化轨迹

for iteration in range(n_iterations):
    # (a) 前向计算：预测值
    y_pred = X_b.dot(theta)

    # (b) 计算损失 MSE = (1/m) * Σ(y_pred - y)²
    cost = (1 / m) * np.sum((y_pred - y) ** 2)
    cost_history.append(cost)

    # (c) 计算梯度  ∂MSE/∂θ = (2/m) * X_bᵀ·(X_b·θ - y)
    gradients = (2 / m) * X_b.T.dot(y_pred - y)

    # (d) 更新参数
    theta = theta - learning_rate * gradients
    theta_history.append(theta.copy())

    # 打印前 5 轮信息
    if iteration < 5:
        print(f"Iter {iteration:>4d} | Cost = {cost:.4f} | θ = {theta.ravel()}")

print("-" * 50)
print(f"最终结果: θ0(截距) = {theta[0][0]:.4f},  θ1(斜率) = {theta[1][0]:.4f}")
print(f"真实期望: θ0 = 4,  θ1 = 3  (接近即可)")
print(f"最终 MSE = {cost_history[-1]:.6f}")

# ---------- 4. 用正规方程验证 ----------
theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f"\n正规方程解: θ0 = {theta_normal[0][0]:.4f},  θ1 = {theta_normal[1][0]:.4f}")

# ============================================================
# 5. 可视化
# ============================================================
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ----- 图1: 数据散点 + 回归直线 -----
ax1 = axes[0]
ax1.scatter(X, y, c='steelblue', alpha=0.6, edgecolors='k', linewidths=0.5, label='样本数据')
X_line = np.array([[0], [2]])
X_line_b = np.c_[np.ones((2, 1)), X_line]
y_line = X_line_b.dot(theta)
ax1.plot(X_line, y_line, 'r-', linewidth=2, label=f'拟合直线: y={theta[0][0]:.2f}+{theta[1][0]:.2f}x')
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('线性回归拟合结果', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# ----- 图2: 损失函数下降曲线 -----
ax2 = axes[1]
ax2.plot(cost_history, color='darkorange', linewidth=1.5)
ax2.set_xlabel('迭代次数', fontsize=12)
ax2.set_ylabel('MSE 损失', fontsize=12)
ax2.set_title('损失函数收敛曲线', fontsize=14)
ax2.grid(True, alpha=0.3)
# 在图上标注最终损失值
ax2.annotate(f'最终 MSE = {cost_history[-1]:.4f}',
             xy=(len(cost_history) - 1, cost_history[-1]),
             xytext=(len(cost_history) * 0.5, max(cost_history) * 0.5),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=11, color='darkorange')

# ----- 图3: 梯度下降过程中回归线的变化 -----
ax3 = axes[2]
ax3.scatter(X, y, c='steelblue', alpha=0.4, edgecolors='k', linewidths=0.3, s=20)
# 选取几个关键迭代展示回归线的演变
show_iters = [0, 1, 2, 5, 10, 50, n_iterations]
colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(show_iters)))
for idx, i in enumerate(show_iters):
    if i < len(theta_history):
        t = theta_history[i]
        y_plot = X_line_b.dot(t)
        label = f'iter {i}' if i < n_iterations else f'iter {i} (最终)'
        ax3.plot(X_line, y_plot, color=colors[idx], linewidth=1.5, label=label)
ax3.set_xlabel('X', fontsize=12)
ax3.set_ylabel('y', fontsize=12)
ax3.set_title('梯度下降过程中拟合线演变', fontsize=14)
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('5_Linear_Regression_Result.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图表已保存到 5_Linear_Regression_Result.png")
