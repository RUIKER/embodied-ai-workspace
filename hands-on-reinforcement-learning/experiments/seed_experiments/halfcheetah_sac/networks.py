"""HalfCheetah SAC 使用的 2×256 actor / critic。

11_SAC/Pendulum 的 GaussianPolicy 只有一层隐藏层；MuJoCo 的标准 SAC 通常用
两层 256。这里保留 SAC 类的接口（action_bias/action_scale/fc_mu），只把网络加深，
这样可以直接复用 11_SAC/Pendulum/agents.py 里的 SAC。
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianPolicy(torch.nn.Module):
    """对角高斯策略 + tanh 压缩，输出落在 [action_low, action_high]。"""

    def __init__(self, state_dim, hidden_dim, action_low, action_high):
        super().__init__()
        action_dim = len(action_low)
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # 网络输出的是“未压缩”的均值；标准差由另一个头给出，和状态相关。
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = torch.nn.Linear(hidden_dim, action_dim)
        # 动作范围可能不对称，所以用 scale/bias 两个 buffer 做仿射变换：
        # a = bias + scale * tanh(u)，tanh(u) ∈ [-1, 1] ⇒ a ∈ [low, high]。
        # 用 register_buffer 而不是普通属性，这样 .to(device) 会自动带走它们。
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer('action_scale', (high - low) / 2)
        self.register_buffer('action_bias', (high + low) / 2)

    def forward(self, states):
        hidden = F.relu(self.fc2(F.relu(self.fc1(states))))
        mean = self.fc_mu(hidden)
        # log_std 夹在 [-20, 2]：下界防止 exp 下溢导致 log_prob = -inf，
        # 上界防止 std 爆炸让采样动作完全随机。
        log_std = torch.clamp(self.fc_log_std(hidden), -20, 2)
        distribution = Normal(mean, log_std.exp())
        # rsample 而不是 sample：重参数化让梯度能穿过采样操作传回 mean/log_std，
        # 这是 SAC actor 能直接对 Q 求梯度的前提。
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        actions = self.action_bias + self.action_scale * squashed
        # tanh 是双射，概率密度要乘雅可比做修正：
        #   log π(a|s) = log μ(u|s) - Σ log(1 - tanh(u)^2) - Σ log(scale)
        # 1e-6 防止 |tanh(u)|→1 时 log(0)。
        correction = torch.log(
            self.action_scale * (1 - squashed.pow(2)) + 1e-6
        )
        log_probs = distribution.log_prob(pre_tanh) - correction
        # 多维动作按维独立，log_prob 相加得到联合对数概率（形状 [B, 1]）。
        return actions, log_probs.sum(dim=-1, keepdim=True)


class Critic(torch.nn.Module):
    """Q(s, a)：输入是状态和动作的拼接，输出一个标量价值。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # 和 actor 不同，critic 必须看到动作，所以第一层输入维度是 state + action。
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        features = torch.cat([states, actions], dim=1)
        features = F.relu(self.fc2(F.relu(self.fc1(features))))
        return self.fc3(features)
