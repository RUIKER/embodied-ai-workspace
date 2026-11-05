"""Mountain Car 状态的标准化与 RBF 特征工程。"""

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


class RBFStateFeaturizer:
    """将二维状态转换为固定长度的随机 RBF 特征。"""

    def __init__(self, observation_space, samples=10_000, components=100,
                 gammas=(5.0, 2.0, 1.0, 0.5), seed=42):
        rng = np.random.default_rng(seed)
        examples = rng.uniform(
            observation_space.low, observation_space.high, size=(samples, 2)
        )
        self.scaler = StandardScaler().fit(examples)
        self.featurizer = FeatureUnion([
            (f"rbf_{gamma}", RBFSampler(
                gamma=gamma, n_components=components, random_state=seed + index
            ))
            for index, gamma in enumerate(gammas)
        ])
        self.featurizer.fit(self.scaler.transform(examples))

    @property
    def feature_count(self):
        """返回特征向量长度。"""
        return len(self.transform([0.0, 0.0]))

    def transform(self, state):
        """把单个 observation 转换为一维 feature vector。"""
        scaled = self.scaler.transform(np.asarray(state, dtype=np.float64).reshape(1, -1))
        return self.featurizer.transform(scaled)[0]