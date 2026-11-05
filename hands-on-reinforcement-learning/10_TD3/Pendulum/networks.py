import torch
import torch.nn.functional as F


class Actor(torch.nn.Module):
    """确定性策略 mu(s)。

    TD3 的 actor 与 DDPG 一样是确定性的：
        state -> tanh -> 映射到 [action_low, action_high]
    探索不靠随机策略，而是在行为动作上额外加高斯噪声。
    """

    def __init__(self, state_dim, hidden_dim, action_low, action_high):
        super().__init__()
        action_dim = len(action_low)
        # 两层隐藏层比 DDPG 示例中的单层更有表达力，也更接近 Spinning Up。
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)

        # 使用 register_buffer 保存动作边界：
        #   - 它们会跟随 model.to(device) 自动迁移到 GPU/CPU；
        #   - 它们不是可训练参数，不会被优化器更新；
        #   - 它们会随 state_dict 一起保存/加载。
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer('action_scale', (high - low) / 2)
        self.register_buffer('action_bias', (high + low) / 2)

    def forward(self, states):
        # tanh 输出 [-1, 1]，再线性映射到环境的实际动作范围。
        # 对 Pendulum：action_low=-2, action_high=2，
        # 因此 bias=0、scale=2，tanh 输出会变成 [-2, 2]。
        hidden = F.relu(self.fc1(states))
        hidden = F.relu(self.fc2(hidden))
        squashed = torch.tanh(self.fc_out(hidden))
        return self.action_bias + self.action_scale * squashed


class Critic(torch.nn.Module):
    """Q(s, a) 网络。

    TD3 会创建两个结构相同但初始化独立（且训练数据顺序相同）的 Critic，
    分别记为 Q1 和 Q2。二者不共享参数，这样才能用 min(Q1, Q2)
    降低单个网络对 Q 值的高估。
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # 状态和动作在输入层拼接，因此第一层输入维度是 state_dim + action_dim。
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        features = torch.cat([states, actions], dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        # 输出形状 [batch_size, 1]，每个状态-动作对一个标量 Q 值。
        return self.fc_out(features)
