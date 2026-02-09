import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV    # 划分数据集，网格搜索交叉验证
from sklearn.compose import ColumnTransformer   # 列转换器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder     # 标准化转换器，独热编码器
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据集
dataset = pd.read_csv('./data/heart_disease.csv')

# 处理缺失值
dataset.dropna(inplace=True)

dataset.info()
print(dataset.head())

# 2. 数据集划分
X = dataset.drop(['是否患有心脏病'], axis=1)
y = dataset["是否患有心脏病"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 3. 特征工程：特征转换
# 数值型特征
numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]
# 类别型特征
categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]
# 二元特征
binary_features = ["性别", "空腹血糖", "运动性心绞痛"]

# 创建列转换器
columnTransformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop="first"), categorical_features),
        ('bin', "passthrough", binary_features),
    ]
)

# 执行特征转换
x_train = columnTransformer.fit_transform(x_train)
x_test = columnTransformer.transform(x_test)

# 1. 准备一个基础模型
# 这里不放任何参数，因为参数稍后由网格搜索来填入
knn = KNeighborsClassifier()

# 2. 定义参数网格 (核心步骤)
# 这是一个字典，key 必须对应 KNN 函数里的参数名
params_grid = {
    'n_neighbors': list(range(1, 11)),   # 试探 K 值：[1, 2, 3, ..., 10]
    'weights': ['uniform', 'distance'],  # 试探权重：[均匀, 距离加权]
    'p': [1, 2]                          # 试探距离：[曼哈顿距离, 欧氏距离]
}
# 这里的组合总数 = 10(K值) * 2(权重) * 2(距离) = 40 种组合

# 3. 初始化网格搜索对象
gs_cv = GridSearchCV(
    estimator=knn,           # 使用哪个模型：KNN
    param_grid=params_grid,  # 要试哪些参数：上面的字典
    cv=10                    # 交叉验证折数：10折 (数据切成10份)
)

# 4. 开始跑代码 (这一步最耗时)
# 它的计算量是：40种参数组合 * 10次交叉验证 = 训练了 400 次模型！
gs_cv.fit(x_train, y_train)

# 5. 查看详细的“考试成绩单”
# cv_results_ 包含了每一种组合的得分、训练时间等详细信息
print(f"网格搜索交叉验证结果:\n{pd.DataFrame(gs_cv.cv_results_).to_string()}")

# 6. 获取“冠军”模型
# best_estimator_ 是已经用最优参数、在所有训练数据上重新训练好的模型
print(f"最优模型: {gs_cv.best_estimator_}") 

# 7. 获取“冠军”参数组合
# 比如它可能会输出：{'n_neighbors': 7, 'p': 1, 'weights': 'distance'}
print(f"最优参数组合: {gs_cv.best_params_}")

# 8. 获取最高分 (这是在验证集上的平均准确率)
print(gs_cv.best_score_)