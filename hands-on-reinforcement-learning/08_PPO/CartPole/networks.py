import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    """离散动作策略网络。

    输入一个状态，输出该状态下每个离散动作的概率。
    例如 CartPole 有两个动作，输出就是 [P(a=0), P(a=1)]。
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        # states: [batch_size, state_dim]
        # logits: [batch_size, action_dim]
        # softmax 后每一行都是一个合法的动作概率分布。
        return F.softmax(self.fc2(F.relu(self.fc1(states))), dim=1)


class ValueNet(torch.nn.Module):
    """状态价值网络，估计 V(s)。

    V(s) 表示从状态 s 出发、继续执行当前策略时的期望回报。
    """

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states):
        # 输出形状为 [batch_size, 1]，每个状态对应一个价值估计。
        return self.fc2(F.relu(self.fc1(states)))
