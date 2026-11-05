import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic
from noise import GaussianNoise, smooth_target_action


class TD3:
    """Twin Delayed DDPG agent。

    与 DDPG 一样，TD3 是 off-policy 的确定性 actor-critic 算法；
    区别在于它维护两个 critic、对 target 动作做平滑，并延迟 actor 更新。

    网络清单（共 6 个）：
        actor            mu(s)
        target_actor     mu_target(s)
        critic_1         Q1(s, a)
        critic_2         Q2(s, a)
        target_critic_1  Q1_target(s, a)
        target_critic_2  Q2_target(s, a)
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_low,
                 action_high, actor_lr, critic_lr, gamma, tau, policy_delay,
                 target_noise, noise_clip, exploration_sigma, device):
        self.actor = Actor(state_dim, hidden_dim, action_low, action_high).to(device)
        self.target_actor = Actor(state_dim, hidden_dim, action_low, action_high).to(device)
        self.critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)

        # target 网络初始时与 online 网络完全相同。
        # 注意：两个 critic 各自独立初始化，不共享参数。
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.device = device

        # 动作边界与半幅，用于探索裁剪和 target smoothing 的单位换算。
        self.action_low = torch.as_tensor(
            action_low, dtype=torch.float32, device=device
        )
        self.action_high = torch.as_tensor(
            action_high, dtype=torch.float32, device=device
        )
        self.action_scale = (self.action_high - self.action_low) / 2

        self.noise = GaussianNoise(action_dim, exploration_sigma)

        # 记录调用 update() 的次数，用于实现 delayed policy update。
        # 注意这个计数是“梯度更新次数”，不是环境步数。
        self.update_count = 0

    def reset_noise(self):
        # 高斯噪声无状态，保留该接口便于和 DDPG 的训练循环保持一致。
        self.noise.reset()

    def take_action(self, state, explore=True):
        """根据状态选择动作。

        explore=True：训练时使用，在确定性动作上叠加高斯噪声。
        explore=False：评估时使用，执行纯 actor 输出。
        """
        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0)

        if explore:
            noise = torch.as_tensor(
                self.noise.sample(), dtype=torch.float32, device=self.device
            )
            action = action + noise

        # 无论是否加噪声，都不能把越界动作交给环境。
        action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action.cpu().numpy()

    def soft_update(self, online_net, target_net):
        """Polyak averaging：target <- (1 - tau) * target + tau * online。"""
        for target_param, online_param in zip(
            target_net.parameters(), online_net.parameters()
        ):
            # 使用 data 原地修改，避免把 target 参数加入计算图。
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * online_param.data)

    def calc_target(self, rewards, next_states, dones):
        """计算 clipped double-Q 的 Bellman target。

        公式：
            a' = clip(mu_target(s') + clip(eps, -c, c), a_low, a_high)
            y  = r + gamma * (1 - done) * min(Q1_target(s', a'),
                                              Q2_target(s', a'))

        取两个 target critic 的最小值，可以理解为对 Q 值高估的保守估计：
        即使某一个网络偶然给出过大的 Q，另一个网络也会把它压下来。
        """
        with torch.no_grad():
            # 第一步：target actor 给出下一状态动作，并做 target policy smoothing。
            next_actions = self.target_actor(next_states)
            next_actions = smooth_target_action(
                next_actions,
                self.target_noise,
                self.noise_clip,
                self.action_scale,
                self.action_low,
                self.action_high,
            )

            # 第二步：两个 target critic 分别打分，取最小值。
            next_q_1 = self.target_critic_1(next_states, next_actions)
            next_q_2 = self.target_critic_2(next_states, next_actions)
            next_q = torch.min(next_q_1, next_q_2)

            # done=1 表示真正终止，不再 bootstrap 下一状态价值；
            # 时间截断（truncated）不在这里体现，trainer 只传 terminated。
            return rewards + self.gamma * next_q * (1 - dones)

    def update(self, batch):
        """执行一次 TD3 更新，返回本次的 loss 信息。

        每次调用都会更新两个 critic；
        只有每 policy_delay 次调用，才额外更新 actor 和三个 target 网络。
        """
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        ).view(-1, 1)

        # ------------------------------------------------------------------
        # 1. 更新两个 critic
        # ------------------------------------------------------------------
        # 两个 critic 回归同一个 target。target 由两个 target critic 的
        # 最小值构成，但每个 online critic 都要向这个 target 靠近。
        q_target = self.calc_target(rewards, next_states, dones)

        q_1 = self.critic_1(states, actions)
        q_2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q_1, q_target) + F.mse_loss(q_2, q_target)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # ------------------------------------------------------------------
        # 2. 延迟更新 actor 和 target 网络
        # ------------------------------------------------------------------
        self.update_count += 1
        actor_loss_value = None

        if self.update_count % self.policy_delay == 0:
            # 策略更新只需要对 actor 参数求梯度。临时冻结 critic，
            # 可以避免在 actor loss 反向传播时计算无用的 critic 梯度，
            # 也防止这些梯度意外残留在 critic 参数上。
            for parameter in self.critic_1.parameters():
                parameter.requires_grad_(False)
            for parameter in self.critic_2.parameters():
                parameter.requires_grad_(False)
            try:
                # 确定性策略梯度：最大化 Q1(s, mu(s))。
                # 优化器只做最小化，因此对 Q 取负号。
                # 注意这里只用 Q1（TD3 论文与 Spinning Up 的做法），
                # 两个 critic 只在计算 Bellman target 时取 min。
                actor_loss = -self.critic_1(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss_value = actor_loss.item()
            finally:
                # 无论反向传播是否异常，都要恢复 critic 的可训练状态。
                for parameter in self.critic_1.parameters():
                    parameter.requires_grad_(True)
                for parameter in self.critic_2.parameters():
                    parameter.requires_grad_(True)

            # actor 更新后，再同步三个 target 网络。
            # 这样 target 变化频率与 actor 一致，比每个 critic step 都同步更稳定。
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_value,
        }
