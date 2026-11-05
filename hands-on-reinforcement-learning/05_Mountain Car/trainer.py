"""训练循环：串联环境、特征工程和 actor-critic 更新。"""

from dataclasses import dataclass
import numpy as np

from environment import reset_environment, step_environment
from environment import MountainCarEnvironment
from estimators import GaussianPolicy, ValueEstimator
from features import RBFStateFeaturizer


@dataclass
class EpisodeStats:
    """记录每个 episode 的总奖励和步数。"""

    episode_rewards: np.ndarray
    episode_lengths: np.ndarray


def train(
    env: MountainCarEnvironment,
    featurizer: RBFStateFeaturizer,
    policy: GaussianPolicy,
    value_estimator: ValueEstimator,
    episodes: int,
    discount_factor: float,
    print_interval: int = 10,
) -> EpisodeStats:
    """执行一步 TD actor-critic 训练，并返回可用于绘图的统计信息。"""
    stats = EpisodeStats(np.zeros(episodes), np.zeros(episodes, dtype=int))
    for episode in range(episodes):
        state = reset_environment(env)
        for step in range(1, env.spec.max_episode_steps + 1):
            features = featurizer.transform(state)
            action = policy.sample(features)
            next_state, reward, terminated, truncated = step_environment(env, np.array([action]))
            done = terminated or truncated
            next_features = featurizer.transform(next_state)
            next_value = value_estimator.predict(next_features)
            # 只有真正终止（到达终点）才没有后续价值；
            # 时间限制截断的状态仍需要 bootstrap。
            target = reward if terminated else reward + discount_factor * next_value
            advantage = target - value_estimator.predict(features)
            value_estimator.update(features, target)
            policy.update(features, action, advantage)
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = step
            state = next_state
            if done:
                break
        if (episode + 1) % print_interval == 0 or episode == 0:
            print(f"Episode {episode + 1:>4}/{episodes}: "
                  f"reward={stats.episode_rewards[episode]:.2f}, "
                  f"steps={stats.episode_lengths[episode]}")
    return stats