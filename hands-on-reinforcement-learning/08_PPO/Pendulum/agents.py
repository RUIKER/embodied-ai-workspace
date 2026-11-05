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
                 reward_shift, reward_scale, device, minibatch_size=64,
                 max_grad_norm=0.5):
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
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
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

        # `dones` 只含真正的 terminated；`ends` 是回合边界（含时间截断）。
        ends = np.asarray(transition_dict['ends'], dtype=np.float32)

        # Pendulum reward 数值较大且为负，缩放只用于 critic 的训练目标；
        # return_list 仍然记录环境原始 reward，方便和教程比较。
        scaled_rewards = (rewards + self.reward_shift) / self.reward_scale
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        td_target = scaled_rewards + self.gamma * next_values * (1 - dones)
        td_delta = td_target - values

        # GAE 在回合边界切断；advantage 与 ratio 都保持 [T, 1]，避免广播成 [T, T]。
        advantage = rl_utils.compute_gae(
            self.gamma, self.lmbda, td_delta, ends
        ).to(self.device).unsqueeze(1)
        # critic 的回归目标是 GAE return = advantage + V(s)，比单步 TD 目标方差小。
        returns = advantage + values
        # old policy 是采样这批数据时的策略；detach 后固定不变。
        old_log_probs = self._log_probs(
            self.actor, states, actions
        ).detach()

        batch_size = states.shape[0]
        for _ in range(self.epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.minibatch_size):
                index = permutation[start:start + self.minibatch_size]
                batch_adv = advantage[index]
                batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
                log_probs = self._log_probs(self.actor, states[index], actions[index])
                # ratio 比较新旧策略对同一个已采样动作的概率。
                ratio = torch.exp(log_probs - old_log_probs[index])
                clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                # PPO-Clip 同时计算未截断目标和截断目标，取更保守的一个。
                surrogate = torch.min(ratio * batch_adv, clipped_ratio * batch_adv)
                actor_loss = -surrogate.mean()
                critic_loss = F.mse_loss(self.critic(states[index]), returns[index])

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()
