import torch
import torch.nn.functional as F


class Actor(torch.nn.Module):
    """确定性策略 mu(s)，输出环境动作范围内的连续动作。"""

    def __init__(self, state_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, len(action_low))
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer('action_scale', (high - low) / 2)
        self.register_buffer('action_bias', (high + low) / 2)

    def forward(self, states):
        # tanh 将输出限制到 [-1, 1]，再映射到环境动作区间。
        hidden = F.relu(self.fc1(states))
        return self.action_bias + self.action_scale * torch.tanh(self.fc2(hidden))


class Critic(torch.nn.Module):
    """Q(s,a) 网络，同时接收状态和动作。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        # 拼接后输入，输出每个状态-动作对的一个标量 Q 值。
        features = torch.cat([states, actions], dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.fc_out(features)
