"""
optimizer.py — 参数更新策略（优化器）
--------------------------------------
遵循"策略模式"：所有优化器继承抽象基类 BaseOptimizer，
只需实现 step() 方法，Trainer 无需关心具体更新逻辑。

当前已实现
----------
- SGD : 随机梯度下降（学习中使用）

预留位置（TODO）
----------------
- Momentum : 动量法
- Adam     : Adam 优化器
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """优化器抽象基类。"""

    @abstractmethod
    def step(self, params: dict, grads: dict) -> None:
        """
        原地更新 params（in-place）。

        params : 模型参数字典 {'W1': ..., 'b1': ..., ...}
        grads  : 对应梯度字典，结构与 params 相同
        """
        ...


# ------------------------------------------------------------------ #
#  SGD — 随机梯度下降
# ------------------------------------------------------------------ #
class SGD(BaseOptimizer):
    """
    随机梯度下降（Stochastic Gradient Descent）。

    更新规则：θ ← θ - lr × ∇θ L
    """

    def __init__(self, learning_rate: float = 0.1):
        self.lr = learning_rate

    def step(self, params: dict, grads: dict) -> None:
        for key in params:
            params[key] -= self.lr * grads[key]


# ------------------------------------------------------------------ #
#  Momentum — 动量法（预留，TODO）
# ------------------------------------------------------------------ #
class Momentum(BaseOptimizer):
    """
    动量法优化器。

    更新规则：
        v ← momentum × v - lr × ∇θ L
        θ ← θ + v

    TODO: 待学习后实现。
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self._velocity: dict = {}

    def step(self, params: dict, grads: dict) -> None:
        raise NotImplementedError("Momentum 优化器尚未实现，请学习后完善。")


# ------------------------------------------------------------------ #
#  Adam — 自适应矩估计（预留，TODO）
# ------------------------------------------------------------------ #
class Adam(BaseOptimizer):
    """
    Adam 优化器（Adaptive Moment Estimation）。

    TODO: 待学习后实现。
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self, params: dict, grads: dict) -> None:
        raise NotImplementedError("Adam 优化器尚未实现，请学习后完善。")


# ------------------------------------------------------------------ #
#  工厂函数
# ------------------------------------------------------------------ #
def build_optimizer(optimizer_name: str, learning_rate: float) -> BaseOptimizer:
    """根据配置名称创建对应优化器实例。"""
    registry = {
        'sgd':      SGD,
        'momentum': Momentum,
        'adam':     Adam,
    }
    name = optimizer_name.lower()
    if name not in registry:
        raise ValueError(f"未知优化器: {optimizer_name!r}，可选: {list(registry.keys())}")
    return registry[name](learning_rate=learning_rate)
