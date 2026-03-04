import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 划分训练集和测试集
from sklearn.preprocessing import PolynomialFeatures    # 构建多项式特征
from sklearn.linear_model import LinearRegression   # 线性回归模型
from sklearn.metrics import mean_squared_error  # 均方误差

plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False

# 1. 构建数据
X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(low=-0.5, high=0.5, size=300).reshape(-1, 1)

# print(X.shape)
# print(y.shape)

# 画出散点图
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
ax[0].scatter(X, y, color='y')
ax[1].scatter(X, y, color='y')
ax[2].scatter(X, y, color='y')
ax[3].scatter(X, y, color='y')

ax[0].set_title("4.1 欠拟合（1次）")
ax[1].set_title("4.2 恰好拟合（5次）")
ax[2].set_title("4.3 过拟合（20次）")
ax[3].set_title("4.4 正则化优化（20次+Ridge）")

# plt.show()

# 2. 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义线性回归模型
model = LinearRegression()

# 4. 分三种情况，分别进行训练和测试

# 4.1 欠拟合（一条直线）
x_train1 = x_train
x_test1 = x_test
model.fit(x_train1, y_train)

# 查看训练完成后的模型参数
print("斜率为: ", model.coef_)
print("截距为：", model.intercept_)

# 画出拟合直线
ax[0].plot(X, model.predict(X), color='r')

# 测试
y_pred1 = model.predict(x_test1)

# 计算误差：训练误差和测试误差
test_loss1 = mean_squared_error(y_test, y_pred1)
train_loss1 = mean_squared_error(y_train, model.predict(x_train1))

ax[0].text(-3, 1, f"测试误差：{test_loss1:.4f}")
ax[0].text(-3, 1.3, f"训练误差：{train_loss1:.4f}")


# 4.2 恰好拟合（5次曲线）
poly5 = PolynomialFeatures(degree=5)

x_train2 = poly5.fit_transform(x_train)
x_test2 = poly5.transform(x_test)
model.fit(x_train2, y_train)

# 查看训练完成后的模型参数
print(model.coef_)
print(model.intercept_)

# 画出拟合曲线
ax[1].plot(X, model.predict( poly5.fit_transform(X) ), color='r')

# 测试
y_pred2 = model.predict(x_test2)

# 计算误差：训练误差和测试误差
test_loss2 = mean_squared_error(y_test, y_pred2)
train_loss2 = mean_squared_error(y_train, model.predict(x_train2))

ax[1].text(-3, 1, f"测试误差：{test_loss2:.4f}")
ax[1].text(-3, 1.3, f"训练误差：{train_loss2:.4f}")

# 4.3 过拟合（20次曲线）
poly20 = PolynomialFeatures(degree=20)

x_train3 = poly20.fit_transform(x_train)
x_test3 = poly20.transform(x_test)
model.fit(x_train3, y_train)

# 查看训练完成后的模型参数
print(model.coef_)
print(model.intercept_)

# 画出拟合曲线
ax[2].plot(X, model.predict( poly20.fit_transform(X) ), color='r')

# 测试
y_pred3 = model.predict(x_test3)

# 计算误差：训练误差和测试误差
test_loss3 = mean_squared_error(y_test, y_pred3)
train_loss3 = mean_squared_error(y_train, model.predict(x_train3))

ax[2].text(-3, 1, f"测试误差：{test_loss3:.4f}")
ax[2].text(-3, 1.3, f"训练误差：{train_loss3:.4f}")

# 4.4 过拟合优化（20次曲线 + 标准化 + Ridge正则化）
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 使用Pipeline：多项式特征 -> 标准化 -> Ridge回归
# 标准化是关键：消除不同次数多项式特征的尺度差异
ridge_pipeline = make_pipeline(
    PolynomialFeatures(degree=20),
    StandardScaler(),
    Ridge(alpha=1.0)  # 标准化后，alpha=1.0 就能有效控制过拟合
)

ridge_pipeline.fit(x_train, y_train)

# 查看Ridge回归系数（需要从pipeline中提取）
print("Ridge回归系数：", ridge_pipeline.named_steps['ridge'].coef_)
print("Ridge回归截距：", ridge_pipeline.named_steps['ridge'].intercept_)

# 画出正则化后的拟合曲线
ax[3].plot(X, ridge_pipeline.predict(X), color='r')

# 测试
y_pred4 = ridge_pipeline.predict(x_test)

# 计算误差：训练误差和测试误差
test_loss4 = mean_squared_error(y_test, y_pred4)
train_loss4 = mean_squared_error(y_train, ridge_pipeline.predict(x_train))

ax[3].text(-3, 1, f"测试误差：{test_loss4:.4f}")
ax[3].text(-3, 1.3, f"训练误差：{train_loss4:.4f}")

plt.tight_layout()
plt.show()
