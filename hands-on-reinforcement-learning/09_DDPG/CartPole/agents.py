import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic
from noise import NormalNoise, OUNoise


class DDPG:
    """教学用车杆DDPG适配器：actor输出连续信号，再映射为左右动作。"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_low,
                 action_high, actor_lr, critic_lr, gamma, tau, noise_type,
                 noise_sigma, ou_theta, device):
        self.actor = Actor(state_dim, hidden_dim, action_low, action_high).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, hidden_dim, action_low, action_high).to(device)
        self.target_critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dim = action_dim

        if noise_type.lower() == 'ou':
            self.noise = OUNoise(action_dim, noise_sigma, ou_theta)
        elif noise_type.lower() == 'normal':
            self.noise = NormalNoise(action_dim, noise_sigma)
        else:
            raise ValueError("noise_type must be 'normal' or 'ou'")

    def reset_noise(self):
        self.noise.reset()

    def take_action(self, state, explore=True):
        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        if explore:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def soft_update(self, online_net, target_net):
        for target_param, online_param in zip(
            target_net.parameters(), online_net.parameters()
        ):
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * online_param.data)

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        critic_loss = F.mse_loss(self.critic(states, actions), q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
