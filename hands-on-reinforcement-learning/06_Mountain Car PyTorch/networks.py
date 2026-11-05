"""Actor and critic networks."""

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Gaussian policy with actions clipped to [-1, 1].

    与 numpy 参考实现一致：直接对高斯采样做 clip，而不是 tanh 压缩。
    clip 会把相当一部分动作饱和到 ±1，形成 bang-bang 式的持续摆动，
    这是小车积累动能爬上山坡、发现 +100 终点奖励的关键；
    tanh 压缩的零均值高斯探索几乎永远无法到达终点，策略会
    收敛到"谷底不动"的局部最优。
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_size, action_size)
        # std=1.0 的初始探索：tanh 压缩后仍有约 30% 的动作接近 ±1，
        # 形成类似 bang-bang 的摆动，才有机会爬上山坡发现 +100 奖励。
        # 初始 std 太小（如 exp(-0.5)≈0.61）会在动作惩罚梯度下迅速塌缩，
        # 策略收敛到"谷底不动"的局部最优。
        self.log_std = nn.Parameter(torch.full((action_size,), 0.0))

    def distribution(self, states: torch.Tensor) -> Normal:
        hidden = self.network(states)
        mean = self.mean_layer(hidden)
        standard_deviation = self.log_std.clamp(-5.0, 1.0).exp()
        return Normal(mean, standard_deviation)

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw_action = distribution.rsample()
        action = raw_action.clamp(-1.0, 1.0)
        # log_prob 用未裁剪的 raw_action（与 numpy 版相同的有偏但有效的近似）。
        log_probability = distribution.log_prob(raw_action).sum(dim=-1)
        return action, log_probability, distribution.entropy().sum(dim=-1)


class Critic(nn.Module):
    """State-value network V(s)."""

    def __init__(self, state_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)
