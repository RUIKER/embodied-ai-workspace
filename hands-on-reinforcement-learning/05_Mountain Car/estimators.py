"""线性 actor 和 critic：只负责预测与参数更新。"""

import numpy as np


class ValueEstimator:
    """使用线性函数近似状态价值 V(s)。"""

    def __init__(self, feature_count, learning_rate=0.1):
        self.weights = np.zeros(feature_count, dtype=np.float64)
        self.learning_rate = learning_rate

    def predict(self, features):
        return float(np.dot(self.weights, features))

    def update(self, features, target):
        prediction = self.predict(features)
        error = target - prediction
        self.weights += self.learning_rate * error * features
        return error


class GaussianPolicy:
    """用线性均值和可学习标准差表示连续动作策略。

    该实现对应参考文档中的 Gaussian policy：初始标准差为
    ``softplus(0)``，并通过 entropy 项保持探索。
    """

    def __init__(self, feature_count, action_low, action_high,
                 learning_rate=0.001, standard_deviation=None,
                 entropy_coefficient=0.1, seed=42):
        self.weights = np.zeros(feature_count, dtype=np.float64)
        initial_deviation = 0.0 if standard_deviation is None else np.log(np.expm1(standard_deviation))
        self.log_standard_deviation_weights = np.full(
            feature_count, initial_deviation, dtype=np.float64
        )
        self.learning_rate = learning_rate
        self.entropy_coefficient = entropy_coefficient
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.rng = np.random.default_rng(seed)

    def mean(self, features):
        return float(np.dot(self.weights, features))

    def log_standard_deviation(self, features):
        return float(np.dot(self.log_standard_deviation_weights, features))

    def standard_deviation(self, features):
        # 与 TensorFlow softplus 保持一致，并限制数值范围避免溢出。
        raw = np.clip(self.log_standard_deviation(features), -10.0, 5.0)
        return float(np.log1p(np.exp(raw)) + 1e-5)

    def sample(self, features):
        action = self.rng.normal(
            self.mean(features), self.standard_deviation(features)
        )
        return float(np.clip(action, self.action_low, self.action_high))

    def update(self, features, action, advantage):
        mean = self.mean(features)
        standard_deviation = self.standard_deviation(features)
        centered_action = action - mean
        mean_gradient = centered_action / standard_deviation ** 2 * features
        log_deviation_gradient = (
            centered_action ** 2 / standard_deviation ** 2 - 1.0
        ) * features
        entropy_gradient = features
        mean_update = self.learning_rate * advantage * mean_gradient
        deviation_update = self.learning_rate * (
            advantage * log_deviation_gradient
            + self.entropy_coefficient * entropy_gradient
        )
        # 线性 Gaussian policy 在连续任务中容易因单次 TD-error 过大而发散。
        # 裁剪只影响数值稳定性，不改变 actor-critic 的更新方向。
        self.weights += np.clip(mean_update, -0.05, 0.05)
        self.log_standard_deviation_weights += np.clip(
            deviation_update, -0.05, 0.05
        )
        self.weights = np.nan_to_num(self.weights, nan=0.0, posinf=10.0, neginf=-10.0)
        self.log_standard_deviation_weights = np.nan_to_num(
            self.log_standard_deviation_weights, nan=0.0, posinf=5.0, neginf=-5.0
        )
        self.weights = np.clip(self.weights, -10.0, 10.0)
        self.log_standard_deviation_weights = np.clip(
            self.log_standard_deviation_weights, -5.0, 5.0
        )