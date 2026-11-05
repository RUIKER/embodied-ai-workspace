import torch
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianPolicy(torch.nn.Module):
    """随机高斯策略。

    网络先产生一个未限制的高斯变量 z，再通过 tanh 和线性缩放
    将动作映射到 Pendulum 的合法动作区间。
    """

    def __init__(self, state_dim, hidden_dim, action_low, action_high):
        super().__init__()
        action_dim = len(action_low)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = torch.nn.Linear(hidden_dim, action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer('action_scale', (high - low) / 2)
        self.register_buffer('action_bias', (high + low) / 2)

    def forward(self, states):
        # states: [batch_size, state_dim]
        hidden = F.relu(self.fc1(states))
        # mean/log_std: [batch_size, action_dim]
        mean = self.fc_mu(hidden)
        log_std = torch.clamp(self.fc_log_std(hidden), -20, 2)
        std = log_std.exp()
        distribution = Normal(mean, std)

        # rsample 使用 z = mean + std * epsilon，epsilon 与网络参数无关，
        # 因此梯度可以穿过随机采样过程回到 mean/std 网络。
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        # 把[-1, 1]映射到环境的[action_low, action_high]。
        actions = self.action_bias + self.action_scale * squashed

        # tanh 和动作缩放会改变概率密度，必须减去变量变换的Jacobian项。
        correction = torch.log(
            self.action_scale * (1 - squashed.pow(2)) + 1e-6
        )
        log_probs = distribution.log_prob(pre_tanh) - correction
        # 多维动作的联合 log probability 等于各维 log probability 之和。
        return actions, log_probs.sum(dim=-1, keepdim=True)


class Critic(torch.nn.Module):
    """Q(s,a) 网络。

    SAC 会实例化两个互相独立的 Critic，使用 min(Q1, Q2)
    减少单个 Q 网络过高估计的影响。
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        # states: [batch_size, state_dim]
        # actions: [batch_size, action_dim]
        features = torch.cat([states, actions], dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        # 输出形状为 [batch_size, 1]。
        return self.fc_out(features)
