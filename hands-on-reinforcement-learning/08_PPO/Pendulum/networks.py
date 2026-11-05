import torch
import torch.nn.functional as F


class PolicyNetContinuous(torch.nn.Module):
    """连续动作策略网络。

    actor 不直接输出一个动作，而是输出 Normal 分布的 mean 和 std，
    再从这个分布中采样动作。这样策略本身仍然是随机策略。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        # 将 tanh 的 [-1, 1] 输出映射到环境动作区间 [low, high]。
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer('action_scale', (action_high - action_low) / 2)
        self.register_buffer('action_bias', (action_high + action_low) / 2)

    def forward(self, states):
        # states: [batch_size, state_dim]
        features = F.relu(self.fc1(states))
        # tanh 保证均值落在动作范围内；softplus 保证标准差为正。
        mu = self.action_bias + self.action_scale * torch.tanh(self.fc_mu(features))
        std = F.softplus(self.fc_std(features)) + 1e-5
        return mu, std


class ValueNet(torch.nn.Module):
    """连续动作 PPO 使用的状态价值网络 V(s)。"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states):
        # 输出 [batch_size, 1]，不依赖具体动作，只评价状态本身。
        return self.fc2(F.relu(self.fc1(states)))
