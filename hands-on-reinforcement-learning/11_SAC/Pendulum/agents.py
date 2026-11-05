import numpy as np
import torch
import torch.nn.functional as F

from networks import Critic, GaussianPolicy


class SAC:
    """连续动作 Soft Actor-Critic agent。

    SAC 同时学习一个随机 actor、两个 online critic 和两个 target critic。
    随机策略的熵被加入目标函数，用于平衡探索和利用。
    """

    def __init__(self, state_dim, hidden_dim, action_low, action_high,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 reward_shift, reward_scale, device):
        self.actor = GaussianPolicy(
            state_dim, hidden_dim, action_low, action_high
        ).to(device)
        # 两个独立 Q 网络，用 min(Q1,Q2) 缓解单个 critic 的过高估计。
        action_dim = len(action_low)
        self.critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # alpha 是熵正则项的权重。优化 log(alpha) 而不是 alpha，
        # 可以保证 alpha=exp(log_alpha)>0。
        self.log_alpha = torch.tensor(
            np.log(0.01), dtype=torch.float32, device=device, requires_grad=True
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.device = device

    def take_action(self, state):
        # 推理时仍从随机策略采样，因此 SAC 不像 DDPG 那样需要外加探索噪声。
        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state)
        return action.squeeze(0).cpu().numpy()

    def calc_target(self, rewards, next_states, dones):
        """计算两个 critic 共同使用的 soft Bellman target。

        y = r + gamma * (min(Q1_target,Q2_target) - alpha*log pi(a'|s'))
        done=1 时不再 bootstrap 下一状态价值。
        """
        with torch.no_grad():
            # 从当前 actor 对下一状态进行重参数化采样。
            next_actions, next_log_probs = self.actor(next_states)
            next_q_1 = self.target_critic_1(next_states, next_actions)
            next_q_2 = self.target_critic_2(next_states, next_actions)
            next_q = torch.min(next_q_1, next_q_2)
            # 因为 log_prob 通常为负，减去 alpha*log_prob 等价于加入熵奖励。
            soft_next_q = next_q - self.log_alpha.exp().detach() * next_log_probs
            return rewards + self.gamma * soft_next_q * (1 - dones)

    def soft_update(self, online_net, target_net):
        """使用 Polyak averaging 缓慢同步 target 网络。"""
        for target_param, online_param in zip(
            target_net.parameters(), online_net.parameters()
        ):
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * online_param.data)

    def update(self, batch):
        # 一个 batch 中的形状通常为：
        # states [B, state_dim]，actions [B, action_dim]，rewards/dones [B, 1]。
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        # 只缩放 critic 的训练目标，return_list 仍记录环境原始回报。
        scaled_rewards = (rewards + self.reward_shift) / self.reward_scale

        # 两个 critic 都回归同一个、无梯度的 soft TD target。
        td_target = self.calc_target(scaled_rewards, next_states, dones)
        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target)
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 重参数化采样让 actor 的梯度可以穿过随机动作到达网络参数。
        new_actions, log_probs = self.actor(states)
        new_q = torch.min(
            self.critic_1(states, new_actions),
            self.critic_2(states, new_actions),
        )
        # actor 同时希望 Q 值高、log_prob 小（熵高）。
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_probs - new_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 自动温度调节：当前熵低于目标时增大 alpha，强化熵奖励；
        # 当前熵高于目标时减小 alpha，让策略更关注回报。
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
