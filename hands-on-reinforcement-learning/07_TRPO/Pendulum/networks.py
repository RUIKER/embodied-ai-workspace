import torch
import torch.nn.functional as F


class PolicyNetContinuous(torch.nn.Module):
    """连续动作策略网络，输出高斯分布的均值和标准差。"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer('action_scale', (action_high - action_low) / 2)
        self.register_buffer('action_bias', (action_high + action_low) / 2)

    def forward(self, states):
        features = F.relu(self.fc1(states))
        mu = self.action_bias + self.action_scale * torch.tanh(self.fc_mu(features))
        std = F.softplus(self.fc_std(features)) + 1e-5
        return mu, std


class ValueNet(torch.nn.Module):
    """状态价值网络，输出V(s)。"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states):
        return self.fc2(F.relu(self.fc1(states)))
