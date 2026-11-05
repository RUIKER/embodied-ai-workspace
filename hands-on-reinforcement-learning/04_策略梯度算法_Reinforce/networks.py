import torch
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    '''定义策略网络，其输入是某个状态，输出则是该状态下的动作概率分布'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1) # 采用在离散动作空间上的softmax()函数来实现一个可学习的多项分布