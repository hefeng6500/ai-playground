"""
model.py — 两层神经网络模型
------------------------------
将网络结构与前向传播逻辑封装为一个清晰的类。

设计说明
--------
- forward()   : 前向传播（已实现）
- loss()      : 计算损失（已实现）
- accuracy()  : 计算准确率（已实现）
- gradient()  : 梯度计算接口（当前使用数值梯度；后续实现反向传播后
                只需替换内部实现，接口保持不变）

激活函数说明
------------
- 隐藏层：Sigmoid（当前）；后续可替换为 ReLU
- 输出层：Softmax（多分类标准选择）

损失函数说明
------------
- 交叉熵损失（Cross Entropy Error）— 多分类标准损失
"""

import numpy as np
from functions import sigmoid, softmax, cross_entropy_error
from config import MODEL_CONFIG


class TwoLayerNet:
    """
    两层全连接神经网络（输入层 → 隐藏层 → 输出层）。

    参数
    ----
    input_size       : 输入特征维度（MNIST = 784）
    hidden_size      : 隐藏层节点数
    output_size      : 输出类别数（MNIST = 10）
    weight_init_std  : 权重初始化标准差（过大→梯度爆炸，过小→梯度消失）
    """

    def __init__(
        self,
        input_size: int  = MODEL_CONFIG['input_size'],
        hidden_size: int = MODEL_CONFIG['hidden_size'],
        output_size: int = MODEL_CONFIG['output_size'],
        weight_init_std: float = MODEL_CONFIG['weight_init_std'],
    ):
        # 参数字典（W1, b1, W2, b2）
        # 命名对应层次：
        #   W1 (input_size × hidden_size) — 第一层权重
        #   b1 (hidden_size,)             — 第一层偏置
        #   W2 (hidden_size × output_size)— 第二层权重
        #   b2 (output_size,)             — 第二层偏置
        self.params: dict = {
            'W1': np.random.randn(input_size, hidden_size) * weight_init_std,
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * weight_init_std,
            'b2': np.zeros(output_size),
        }

    # ---------------------------------------------------------------- #
    #  前向传播
    # ---------------------------------------------------------------- #
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播，返回各类别的预测概率分布。

        X : shape (n, input_size) — n 个样本
        返回 : shape (n, output_size) — 每个样本属于各类别的概率
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 第一层：线性变换 + Sigmoid 激活
        a1 = X @ W1 + b1        # (n, hidden_size)
        z1 = sigmoid(a1)        # (n, hidden_size)

        # 第二层：线性变换 + Softmax 输出概率
        a2 = z1 @ W2 + b2       # (n, output_size)
        y  = softmax(a2)        # (n, output_size)

        return y

    # ---------------------------------------------------------------- #
    #  损失计算
    # ---------------------------------------------------------------- #
    def loss(self, X: np.ndarray, t: np.ndarray) -> float:
        """
        计算交叉熵损失。

        X : shape (n, input_size)  — 输入特征
        t : shape (n,)             — 真实标签（整数编码）
        返回 : scalar 损失值
        """
        y = self.forward(X)
        return cross_entropy_error(y, t)

    # ---------------------------------------------------------------- #
    #  准确率计算
    # ---------------------------------------------------------------- #
    def accuracy(self, X: np.ndarray, t: np.ndarray) -> float:
        """
        计算分类准确率。

        X : shape (n, input_size)
        t : shape (n,) — 整数标签
        返回 : float，正确预测比例 [0, 1]
        """
        y_proba = self.forward(X)
        y_pred  = np.argmax(y_proba, axis=1)   # 取概率最大的类别
        return float(np.mean(y_pred == t))

    # ---------------------------------------------------------------- #
    #  梯度计算接口（当前：数值梯度；预留反向传播替换入口）
    # ---------------------------------------------------------------- #
    def gradient(self, X: np.ndarray, t: np.ndarray, method: str = 'numerical') -> dict:
        """
        计算所有参数的梯度，返回与 self.params 结构相同的字典。

        参数
        ----
        X      : 输入特征
        t      : 真实标签
        method : 梯度计算方式
                 - 'numerical'  数值梯度（当前实现，慢但无需推导）
                 - 'backprop'   反向传播（TODO：学习反向传播后实现）

        返回
        ----
        grads : dict，键与 self.params 相同，值为对应梯度数组
        """
        if method == 'numerical':
            return self._numerical_gradient(X, t)
        elif method == 'backprop':
            return self._backprop_gradient(X, t)
        else:
            raise ValueError(f"未知的梯度计算方法: {method!r}，可选 'numerical' / 'backprop'")

    def _numerical_gradient(self, X: np.ndarray, t: np.ndarray) -> dict:
        """数值梯度（中心差分法）——当前实现。"""
        from gradient import numerical_gradient as _num_grad

        # 把损失函数包装为只关于参数的函数
        loss_fn = lambda _: self.loss(X, t)

        grads = {}
        for key in self.params:
            grads[key] = _num_grad(loss_fn, self.params[key])
        return grads

    def _backprop_gradient(self, X: np.ndarray, t: np.ndarray) -> dict:
        """
        反向传播梯度（解析梯度）。

        TODO: 待学习反向传播后实现。
              实现后将比数值梯度快数倍，训练速度大幅提升。
        """
        raise NotImplementedError(
            "反向传播尚未实现，请学习反向传播后完善此方法。\n"
            "当前请使用 method='numerical'（数值梯度）。"
        )
