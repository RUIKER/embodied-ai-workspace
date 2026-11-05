"""Episode-batch actor-critic training with n-step returns (PyTorch autograd).

更新方式：每个 episode 收集完整轨迹，episode 结束后用 n 步 TD 回报
做一次批量更新。相比逐步（per-step）更新：

1. n 步回报把终点 +100 的信号一次回传 n 步，信用传播快得多——
   逐步一步 TD 每个状态要等下次访问才能传播价值，MLP critic 用
   Adam 学习时价值传播极慢，200 个 episode 内 advantage 基本不含
   位置信息，策略均值学不到"该往哪边摆"。
2. 批量平均显著降低梯度方差。逐步更新时，Adam 的二阶矩归一化会把
   大量小 advantage 步骤的噪声梯度放大到与少数关键步骤同等量级，
   策略被噪声主导；同时"到达终点的都是极端动作"这一事实会把
   log_std 一步步顶到上限（棘轮效应），行为彻底被探索噪声淹没。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from environment import reset_environment, step_environment
from networks import Actor, Critic


@dataclass
class EpisodeStats:
    rewards: np.ndarray
    lengths: np.ndarray


def normalize_state(state: np.ndarray, observation_space: Any) -> torch.Tensor:
    """Map the bounded environment observation approximately to [-1, 1]."""
    low = np.asarray(observation_space.low, dtype=np.float32)
    high = np.asarray(observation_space.high, dtype=np.float32)
    normalized = 2.0 * (state - low) / (high - low) - 1.0
    return torch.as_tensor(normalized, dtype=torch.float32)


def n_step_targets(
    rewards: list,
    terminateds: list,
    next_values: np.ndarray,
    n_step: int,
    discount_factor: float,
    reward_scale: float,
) -> torch.Tensor:
    """Compute the n-step TD target of every step in the episode.

    只有真正终止（terminated，到达终点）才停止 bootstrap；
    时间限制截断（truncated）时用最后状态的 V 继续 bootstrap。
    """
    episode_length = len(rewards)
    targets = np.zeros(episode_length, dtype=np.float32)
    for t in range(episode_length):
        accumulated, discount = 0.0, 1.0
        last = min(t + n_step, episode_length) - 1
        for i in range(t, last + 1):
            accumulated += discount * rewards[i] * reward_scale
            discount *= discount_factor
            if terminateds[i]:
                break
        else:
            accumulated += discount * next_values[last]
        targets[t] = accumulated
    return torch.as_tensor(targets)


def train(
    environment: Any,
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    episodes: int,
    discount_factor: float,
    entropy_coefficient: float,
    n_step: int,
    reward_scale: float,
    print_interval: int = 10,
) -> EpisodeStats:
    """Train actor and critic with episode-batch n-step TD updates."""
    stats = EpisodeStats(np.zeros(episodes, dtype=np.float32), np.zeros(episodes, dtype=int))
    observation_space = environment.observation_space
    for episode in range(episodes):
        state = reset_environment(environment)
        states, actions, rewards, terminateds, next_states = [], [], [], [], []
        for step in range(1, environment.spec.max_episode_steps + 1):
            state_tensor = normalize_state(state, observation_space).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _ = actor.sample(state_tensor)
            action = action_tensor.cpu().numpy()[0]
            next_state, reward, terminated, truncated = step_environment(
                environment, action
            )
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            terminateds.append(terminated)
            next_states.append(next_state)
            stats.rewards[episode] += reward
            stats.lengths[episode] = step
            state = next_state
            if terminated or truncated:
                break

        states_tensor = torch.stack(
            [normalize_state(s, observation_space) for s in states]
        )
        actions_tensor = torch.as_tensor(np.stack(actions), dtype=torch.float32)
        next_states_tensor = torch.stack(
            [normalize_state(s, observation_space) for s in next_states]
        )
        with torch.no_grad():
            next_values = critic(next_states_tensor).numpy()
        targets = n_step_targets(
            rewards, terminateds, next_values, n_step, discount_factor, reward_scale
        )

        # reward_scale 让 critic 的目标值保持在 O(1)（终点 +100 -> +1），
        # MLP+Adam 才能快速拟合；统计输出仍用原始奖励，曲线单位不变。
        values = critic(states_tensor)
        critic_loss = (targets - values).pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()

        distribution = actor.distribution(states_tensor)
        # 与 numpy 参考实现一致：直接对 clip 后的动作求高斯 log_prob
        #（有偏但简单有效的近似）。
        log_probability = distribution.log_prob(actions_tensor).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        advantages = (targets - values).detach()
        actor_loss = -(log_probability * advantages).mean()
        actor_optimizer.zero_grad()
        (actor_loss - entropy_coefficient * entropy.mean()).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optimizer.step()

        if (episode + 1) % print_interval == 0 or episode == 0:
            print(
                f"Episode {episode + 1:>4}/{episodes}: "
                f"reward={stats.rewards[episode]:.2f}, "
                f"steps={stats.lengths[episode]}"
            )
    return stats
