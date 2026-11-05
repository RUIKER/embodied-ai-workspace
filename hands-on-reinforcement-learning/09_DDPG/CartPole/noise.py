import numpy as np


class NormalNoise:
    """每一步独立采样的高斯噪声。"""

    def __init__(self, action_dim, sigma):
        self.action_dim = action_dim
        self.sigma = sigma

    def reset(self):
        pass

    def sample(self):
        return self.sigma * np.random.randn(self.action_dim)


class OUNoise:
    """带时间相关性的 Ornstein-Uhlenbeck 噪声。"""

    def __init__(self, action_dim, sigma, theta=0.15, mu=0.0):
        self.action_dim = action_dim
        self.sigma = sigma
        self.theta = theta
        self.mu = mu
        self.state = np.zeros(action_dim, dtype=np.float32)

    def reset(self):
        self.state.fill(0.0)

    def sample(self):
        random_part = self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + self.theta * (self.mu - self.state) + random_part
        return self.state.copy()
