import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    """离散动作策略网络，输出每个动作的概率。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        features = F.relu(self.fc1(states))
        return F.softmax(self.fc2(features), dim=1)


class ValueNet(torch.nn.Module):
    """状态价值网络，输出V(s)。"""

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states):
        features = F.relu(self.fc1(states))
        return self.fc2(features)
