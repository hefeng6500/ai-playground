"""
functions.py — 激活函数与损失函数
-----------------------------------
从 common/functions.py 迁移至本目录，供 test 包内部独立使用，
避免对上层 common 模块产生路径依赖。
"""

import numpy as np


# ================================================================== #
#  激活函数
# ================================================================== #

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 激活函数：将输入压缩到 (0, 1)。"""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活函数：保留正值，负值置零。"""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax 函数：将向量转换为概率分布（各分量之和为 1）。
    支持 1D（单样本）和 2D（批量）输入，内置溢出防护。
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)      # 防止 exp 溢出
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# ================================================================== #
#  损失函数
# ================================================================== #

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    交叉熵损失（Cross Entropy Error）。

    参数
    ----
    y : 预测概率，shape (n, num_classes)
    t : 真实标签，shape (n,)（整数标签）或 (n, num_classes)（one-hot）

    返回
    ----
    scalar — 平均交叉熵损失
    """
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    # 如果 t 是 one-hot 编码，转换为整数标签
    if t.size == y.size:
        t = np.argmax(t, axis=1)
    n = y.shape[0]
    # 取出正确类别的预测概率并计算负对数，加 1e-7 防止 log(0)
    return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n
