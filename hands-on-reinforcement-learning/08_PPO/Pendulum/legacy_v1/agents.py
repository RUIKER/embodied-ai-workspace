import numpy as np
import torch
import torch.nn.functional as F

import rl_utils
from networks import PolicyNetContinuous, ValueNet


class PPOContinuous:
    """连续动作 PPO。

    与 CartPole 版本共享 PPO 的 clipped objective 和 GAE，
    但 actor 使用 Normal 分布，动作和 log probability 都是浮点向量。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_low,
                 action_high, actor_lr, critic_lr, lmbda, epochs, eps, gamma,
                 reward_shift, reward_scale, device):
        self.actor = PolicyNetContinuous(
            state_dim, hidden_dim, action_dim, action_low, action_high
        ).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        self.action_low = torch.as_tensor(
            action_low, dtype=torch.float32, device=device
        )
        self.action_high = torch.as_tensor(
            action_high, dtype=torch.float32, device=device
        )
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.device = device

    @staticmethod
    def _distribution(actor, states):
        # mean/std 的形状都是 [batch_size, action_dim]。
        mean, std = actor(states)
        return torch.distributions.Normal(mean, std)

    def take_action(self, state):
        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        # 保存策略实际采样动作，不能先 clamp 后再计算 log probability，
        # 否则 old_log_probs 与实际采样分布不一致，PPO ratio 会失真。
        action = self._distribution(self.actor, state).sample()
        # 保留策略实际采样的动作，保证old_log_probs与动作一致。
        # Pendulum环境会在执行时将动作限制到合法范围。
        return action.squeeze(0).detach().cpu().numpy()

    @staticmethod
    def _log_probs(actor, states, actions):
        # Normal 对每个动作维度分别给出 log_prob；联合动作概率是各维度之和。
        log_probs = PPOContinuous._distribution(actor, states).log_prob(actions)
        return log_probs.sum(dim=-1, keepdim=True)

    def update(self, transition_dict):
        # 连续动作 actions 的形状通常为 [T, action_dim]，不能按离散动作转成long。
        states = torch.as_tensor(
            np.asarray(transition_dict['states']),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray(transition_dict['actions']),
            dtype=torch.float32,
            device=self.device,
        )
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

        # Pendulum reward 数值较大且为负，缩放只用于 critic 的训练目标；
        # return_list 仍然记录环境原始 reward，方便和教程比较。
        scaled_rewards = (rewards + self.reward_shift) / self.reward_scale
        td_target = scaled_rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        # advantage 与 ratio 都保持 [T, 1]，避免广播成 [T, T]。
        advantage = rl_utils.compute_advantage(
            self.gamma, self.lmbda, td_delta
        ).to(self.device).unsqueeze(1)
        # old policy 是采样这批数据时的策略；detach 后固定不变。
        old_log_probs = self._log_probs(
            self.actor, states, actions
        ).detach()

        for _ in range(self.epochs):
            log_probs = self._log_probs(self.actor, states, actions)
            # ratio 比较新旧策略对同一个已采样动作的概率。
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            # PPO-Clip 同时计算未截断目标和截断目标，取更保守的一个。
            surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)
            actor_loss = -surrogate.mean()
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
