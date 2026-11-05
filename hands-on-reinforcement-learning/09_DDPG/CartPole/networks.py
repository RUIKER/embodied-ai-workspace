import torch
import torch.nn.functional as F


class Actor(torch.nn.Module):
    """输出[-1, 1]连续信号的确定性策略。"""

    def __init__(self, state_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, len(action_low))
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer('action_scale', (high - low) / 2)
        self.register_buffer('action_bias', (high + low) / 2)

    def forward(self, states):
        hidden = F.relu(self.fc1(states))
        return self.action_bias + self.action_scale * torch.tanh(self.fc2(hidden))


class Critic(torch.nn.Module):
    """将状态和连续动作拼接后估计Q(s,a)。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        features = torch.cat([states, actions], dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.fc_out(features)
