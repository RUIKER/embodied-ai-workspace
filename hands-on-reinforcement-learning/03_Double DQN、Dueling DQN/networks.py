import torch
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    '''普通DQN网络：输入状态，输出每个离散动作的Q值。'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 输入形状通常是[batch_size, state_dim]，输出形状是[batch_size, action_dim]。
        # 输出的每一列对应一个动作的Q值。
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DuelingQnet(torch.nn.Module):
    '''Dueling DQN网络：分别估计状态价值V(s)和动作优势A(s,a)。'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DuelingQnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = F.relu(self.fc1(x))
        advantage = self.fc_A(features)
        value = self.fc_V(features)

        # Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))。
        # 减去平均优势值，可以消除V和A之间的表示歧义。
        return value + advantage - advantage.mean(dim=1, keepdim=True)