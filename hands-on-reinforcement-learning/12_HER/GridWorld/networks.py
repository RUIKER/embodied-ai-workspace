"""HER 实验用的 actor / critic。

和 09_DDPG/Pendulum/networks.py 结构一致，只有输入维度从 3 变成 4：
状态是 [x, y, goal_x, goal_y]，动作是二维平面位移。

======================================================================
一个值得记住的点：网络并不知道哪两维是目标
======================================================================
它只是多看到两个数，目标信息通过输入维度"混"进特征里。
也就是说，"目标导向"不是靠网络结构实现的，而是靠**训练分布**——
HER 让同一批 (s, a, s') 反复配上不同的假目标 g'，
网络才从数据里学到 Q(s, a, g) 对 g 的依赖。

（更讲究的做法是 goal-conditioned 结构：把 g 单独过一层再和 s 拼接，
或者用 attention 让策略显式关注目标。这个 5×5 的小任务上没有必要，
但换到机械臂任务时，这种结构能明显提升样本效率。）

======================================================================
形状（B = batch size）
======================================================================
    states        (B, 4)
    actions       (B, 2)
    Actor   μ:    4 → 128 → 128 → 2，输出 tanh 后再乘 action_bound
    Critic  Q:    (4+2)=6 → 128 → 128 → 1，输出是 Q 值，不加激活

两个网络都是两层隐藏层的 MLP：这个任务的状态是低维连续向量，
不需要卷积或循环结构，加深只会让稀疏奖励下的训练更难。
"""

import torch
import torch.nn.functional as F


class Actor(torch.nn.Module):
    """确定性策略 μ(s)：输入 [x, y, gx, gy]，输出二维动作。

    DDPG 的 actor 是"确定性"的：给定 s，输出唯一的 a（不是动作分布）。
    探索靠外部加噪声（见 agents.take_action），而不是靠采样。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        """Args:
            state_dim:   观测维度，这里是 4。
            hidden_dim:  隐藏层宽度，默认 128。
            action_dim:  动作维度，这里是 2。
            action_bound: 动作上限，输出会乘上它。
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)     # (B, 4)   -> (B, 128)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)    # (B, 128) -> (B, 128)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)    # (B, 128) -> (B, 2)
        self.action_bound = float(action_bound)

    def forward(self, states):
        # ① 两层 ReLU 提特征。目标只是输入里的两个数，
        #    网络需要自己学出"目标位置与当前位置的相对关系"。
        hidden = F.relu(self.fc1(states))
        hidden = F.relu(self.fc2(hidden))

        # ② tanh 把输出压到 [-1, 1]，再乘以动作上限。
        #    这样 μ(s) 本身天然落在合法动作范围内，网络不会输出"物理上不可能"的动作。
        #    但探索噪声是在这一步之后加的，仍可能越界，
        #    所以 agents.take_action 里还要再 clip 一次。
        return torch.tanh(self.fc3(hidden)) * self.action_bound


class Critic(torch.nn.Module):
    """Q(s, a)：把状态（含目标）和动作拼在一起，输出一个标量。

    critic 是 DDPG 里唯一能"评价动作好坏"的网络：
    actor 更新时要用 ∂Q/∂a，所以 Q 必须对动作可微（而不是像 DQN 那样离散取 max）。
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # 输入维度是 state_dim + action_dim：状态和动作在**第一层之前**就拼接，
        # 之后再一起过全连接。这是最直接的做法；
        # 也可以做成 twin Q（TD3）或 Q(s) 与 Q(s,a) 分离（Dueling），这里不需要。
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        # (B, 4) 和 (B, 2) 沿特征维拼成 (B, 6)
        features = torch.cat([states, actions], dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        # 输出层不加激活：Q 值可正可负，这里奖励是 0 / −1，Q 理论上在 [−50, 0] 附近。
        return self.fc3(features)
