import numpy as np


class NormalNoise:
    """独立高斯探索噪声：每一步重新采样。"""

    def __init__(self, action_dim, sigma):
        self.action_dim = action_dim
        self.sigma = sigma

    def reset(self):
        # 高斯噪声没有跨时间状态，reset 只是统一接口。
        pass

    def sample(self):
        return self.sigma * np.random.randn(self.action_dim)


class OUNoise:
    """Ornstein-Uhlenbeck 噪声，具有时间相关性。

    x_t = x_{t-1} + theta * (mu - x_{t-1}) + sigma * N(0, I)
    偏离均值后会受到拉回均值的漂移，适合有惯性的连续控制环境。
    """

    def __init__(self, action_dim, sigma, theta=0.15, mu=0.0):
        self.action_dim = action_dim
        self.sigma = sigma
        self.theta = theta
        self.mu = mu
        self.state = np.zeros(action_dim, dtype=np.float32)

    def reset(self):
        # 每个 episode 开始时清除上一条轨迹的噪声状态。
        self.state.fill(0.0)

    def sample(self):
        random_part = self.sigma * np.random.randn(self.action_dim)
        drift = self.theta * (self.mu - self.state)
        self.state = self.state + drift + random_part
        return self.state.copy()
