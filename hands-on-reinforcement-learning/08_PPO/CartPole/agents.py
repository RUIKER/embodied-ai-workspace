import numpy as np
import torch
import torch.nn.functional as F

import rl_utils
from networks import PolicyNet, ValueNet


class PPO:
    """离散动作 PPO。

    actor 学习策略 pi(a|s)，critic 学习状态价值 V(s)。
    PPO 使用旧策略采样的数据重复训练多个 epoch，
    但通过 clip 限制新旧策略变化不能过大。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        # 单个状态先扩展成 batch，形状从 [state_dim] 变为 [1, state_dim]。
        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        # actor 输出概率，Categorical 按概率随机采样离散动作。
        action_dist = torch.distributions.Categorical(self.actor(state))
        return action_dist.sample().item()

    @staticmethod
    def _log_probs(actor, states, actions):
        # actions 原本是 [batch_size, 1]，Categorical 要求 [batch_size]。
        # 最后再变回 [batch_size, 1]，与 advantage 和 ratio 对齐。
        action_dist = torch.distributions.Categorical(actor(states))
        return action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)

    def update(self, transition_dict):
        # 下面所有 batch 的第一维都是 T，即一个 episode 的时间步数量。
        states = torch.as_tensor(
            np.asarray(transition_dict['states']),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            transition_dict['actions'], dtype=torch.long, device=self.device
        ).view(-1, 1)
        rewards = torch.as_tensor(
            transition_dict['rewards'],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        next_states = torch.as_tensor(
            np.asarray(transition_dict['next_states']),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            transition_dict['dones'],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        # TD target：critic 应该拟合的监督信号。
        # done=1 时没有下一状态价值，避免把终止后的价值继续 bootstrap。
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # GAE 将多个时间步的 TD error 加权组合，降低策略梯度方差。
        # compute_advantage 返回 [T]，unsqueeze 后变成 [T, 1]。
        advantage = rl_utils.compute_advantage(
            self.gamma, self.lmbda, td_delta
        ).to(self.device).unsqueeze(1)
        # 重要性采样需要记录“采样时的旧策略概率”。
        # detach 后，旧概率在多个 epoch 中保持常数，不参与反向传播。
        old_log_probs = self._log_probs(
            self.actor, states, actions
        ).detach()

        for _ in range(self.epochs):
            # 用同一批旧数据重新计算当前 actor 的动作概率。
            log_probs = self._log_probs(self.actor, states, actions)

            # ratio = pi_new(a|s) / pi_old(a|s)。
            # ratio > 1 表示新策略提高了该动作的概率，反之则降低。
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)

            # 当 ratio 变化超过 [1-eps, 1+eps] 时使用截断值，
            # 防止一次更新让策略变化过大。
            surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)
            actor_loss = -surrogate.mean()

            # critic 是普通回归问题，拟合 td_target。
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
