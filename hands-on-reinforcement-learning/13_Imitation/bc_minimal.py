"""行为克隆：把专家走过的状态-动作对当成监督数据，回归一个策略。

这个脚本刻意做得小：一个规则专家、一个两层 MLP、三个数字、一条曲线。
目的是看清 BC 的三种失败方式，而不是刷分。

======================================================================
一、BC 在做什么
======================================================================
专家的轨迹给出数据集 D = {(s_i, a_i)}，训练目标是最大化专家动作的似然：

    θ* = argmax  Σ_i log π_θ(a_i | s_i)

π_θ 是确定性高斯、方差固定时，上式等价于最小化均方误差：

    L(θ) = E_(s,a)~D ‖π_θ(s) − a‖²

所以代码里的 MSE 不是“随手选的损失”，它就是 MLE 在固定方差下的形式。
换成多峰专家和单峰高斯的 MSE，代价会在第二节出现。

======================================================================
二、三个数字各自测什么
======================================================================
1. 训练 MSE（分布内）
   在示范状态上算动作误差。它低，只说明“专家见过的状态上做得对”。

2. 闭环成功率（分布外）
   自己跑 rollout，状态由策略自己产生。一旦偏离示范分布，
   误差会改变下一个状态，再放大下一个误差——这是 compounding error，
   BC 最核心的问题，DAgger 论文里的 O(T²ε) 误差界说的就是它。

3. 偏移起点成功率（分布偏移的探针）
   把起点从原点挪开几十厘米再跑同一个策略。示范里没有这些状态，
   成功率会明显掉。这是把“分布偏移”从一个说法变成一次测量。

======================================================================
三、两个任务，测两件不同的事
======================================================================
--task grid     GridWorld：专家 a = g − p 是状态的反馈律，BC 学到的也是反馈律，
                误差会自我纠正，所以 5 条示范也能到 100%。这条曲线是平的。
                它说明“示范条数—成功率”不是任何任务都成立的规律——
                取决于任务本身对误差是否宽容，不取决于 BC 这个算法。
--task corridor 两段走廊：专家随机走上绕或下绕，两条路代价完全相同。
                单峰高斯无法表示“要么 +1 要么 −1”，MSE 把两者平均成 0，
                策略直着撞墙。示范条数加到 200 条也救不回来，因为这是
                表示能力问题，不是数据量问题——这正是 ACT（CVAE 引入 latent）
                和 Diffusion Policy 存在的理由。

======================================================================
四、这一版没有的东西
======================================================================
- 没有 DAgger 的在线聚合，只有离线示范。
- 专家是规则策略，不是训练出来的。换成 DDPG+HER 权重时，
  只需要把 Expert.__call__ 换成一个网络前向，其余不动。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 复用 12_HER 的环境，保证 BC 与 HER 那一章用的是同一套动力学和成功判定。
GRIDWORLD = Path(__file__).resolve().parents[1] / '12_HER' / 'GridWorld'
sys.path.insert(0, str(GRIDWORLD))
from environment import GridWorldEnv  # noqa: E402


class Expert:
    """走直线的规则专家：a = clip(g − p, −1, 1)。

    5×5 平面、每步每维最多移动 1 个单位，所以这个专家在 4~5 步内必到。
    它几乎就是最优策略，因此下面所有“BC 比专家差多少”都来自 BC 本身，
    而不是专家不够好。
    """

    def __call__(self, obs):
        return np.clip(obs[2:] - obs[:2], -1.0, 1.0)


def collect(env, expert, n_episodes, seed=0, action_noise=0.0):
    """跑 n_episodes 条专家轨迹，返回 (states, actions, returns)。

    action_noise 模拟“专家不是完美演示”：策略在专家动作上叠高斯噪声后再执行，
    但示范里存的仍然是**加了噪声的动作**。真实数据总是这样——
    人会手抖，RL 策略带探索噪声，遥操作有延迟。
    """
    rng = np.random.default_rng(seed)
    states, actions, returns = [], [], []
    for _ in range(n_episodes):
        if hasattr(expert, 'reset'):
            expert.reset()
        obs, _ = env.reset()
        ep_r = 0.0
        while True:
            a = expert(obs) + action_noise * rng.standard_normal(env.action_dim)
            a = np.clip(a, -1.0, 1.0)
            states.append(obs.copy())
            actions.append(a.copy())
            obs, r, terminated, truncated, _ = env.step(a)
            ep_r += r
            if terminated or truncated:
                break
        returns.append(ep_r)
    return (np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(returns, dtype=np.float32))


class BCPolicy(nn.Module):
    """4 → 128 → 128 → 2，输出过 tanh 落到 [−1, 1]。

    动作本来就归一化在 [−1, 1]，用 tanh 是为了让策略在边界上不越界；
    代价是边界处的梯度会变小（tanh 饱和），离目标很远时输出被压住——
    这个任务里影响不大，换成关节力矩就得换成无界输出 + clip。
    """

    def __init__(self, obs_dim=4, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def train_bc(states, actions, epochs=300, batch_size=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    policy = BCPolicy(states.shape[1], actions.shape[1])
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.from_numpy(states)
    Y = torch.from_numpy(actions)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(policy(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        mse = float(loss_fn(policy(X), Y))
    return policy, mse



class CorridorEnv:
    """两段走廊：起点 (0, 2.5)，终点 (4.5, 2.5)，中间一堵墙，上下各留一个缺口。

    ┌───────────┐
    │           │   y 方向缺口在墙两侧
    │     ▓▓▓   │
    │ ▓▓▓▓▓▓▓▓▓ │   x = 2.5, y ∈ [1.5, 3.5] 是墙
    │     ▓▓▓   │
    │           │
    └───────────┘

    这个任务是为了单独隔离出“多模态被平均”这一个现象：
    专家每回合随机选从上面绕或从下面绕，两条路**代价完全相同**，
    所以最优点不唯一；而 MSE 对应的单峰高斯只有一个峰，
    它只能把 +1 和 −1 平均成 0，也就是直着撞墙。

    GridWorld 那个任务做不到这一点：目标在右上，专家几乎只有一条路，
    而且 a = g − p 是一个反馈律，BC 学到的也是反馈律，误差会自我纠正。
    """

    size = 5.0
    action_dim = 2
    observation_dim = 4
    start = np.array([0.0, 2.5])
    goal = np.array([4.5, 2.5])
    WALL_X = 2.5
    WALL_LOW, WALL_HIGH = 1.5, 3.5

    def __init__(self, max_steps=60, threshold=0.15):
        self.max_steps = max_steps
        self.threshold = threshold
        self.action_bound = 1.0
        self.rng = np.random.default_rng(0)
        self.state = self.start.copy()
        self.count = 0

    def _observation(self):
        return np.concatenate([self.state, self.goal]).astype(np.float64)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start.copy()
        self.count = 0
        return self._observation(), {}

    def test_set_goal(self, goal=None):
        self.state = self.start.copy()
        self.count = 0
        return self._observation(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        new = np.clip(self.state + a, 0.0, self.size)
        # 穿墙判定：x 从墙的一侧跨到另一侧，且 y 落在墙的范围内 → 只截住 x
        if (self.state[0] - self.WALL_X) * (new[0] - self.WALL_X) < 0 \
                and self.WALL_LOW < self.state[1] < self.WALL_HIGH:
            new[0] = self.state[0]
        self.state = new
        self.count += 1
        reached = np.linalg.norm(self.state - self.goal) <= self.threshold
        return self._observation(), (-1.0 if not reached else 0.0), reached, \
               self.count >= self.max_steps, {}


class CorridorExpert:
    """两段路：先走到缺口 (2.5, 4.2) 或 (2.5, 0.8)，再朝目标走。

    每回合开头随机选上/下，两个分支的回报完全相同。
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.side = 1

    def reset(self):
        self.side = 1 if self.rng.random() < 0.5 else -1

    def __call__(self, obs):
        pos = obs[:2]
        wy = 4.2 if self.side > 0 else 0.8
        waypoint = np.array([2.5, wy])
        # 还没过墙就走缺口，过了墙直奔目标
        target = waypoint if pos[0] < self.WALL_X - 0.05 else self.goal
        d = target - pos
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return np.zeros(2)
        # 步长不超过剩余距离，否则会在目标附近来回振荡（1.0 的步长对 0.15 的到达半径太大）
        return d / n * min(1.0, n)


CorridorExpert.WALL_X = CorridorEnv.WALL_X
CorridorExpert.goal = CorridorEnv.goal
@torch.no_grad()
def bc_act(policy, obs):
    return policy(torch.from_numpy(obs[None].astype(np.float32)))[0].numpy()


def evaluate(act_fn, env, goals=None, start_radius=0.0, seed=0):
    """跑闭环 rollout，返回成功率。

    act_fn(obs) -> action。传 BC 策略或专家都行，两者用同一套评测。
    goals 为 None 时只跑一回合（Corridor 任务目标固定）；否则对每个目标各跑一回合。
    start_radius > 0 时把起点挪到原点附近该半径内，用来测分布偏移。
    """
    rng = np.random.default_rng(seed)
    goals = [None] if goals is None else goals
    wins = 0
    for g in goals:
        env.test_set_goal(g) if g is not None else env.test_set_goal()
        if start_radius > 0:
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, start_radius)
            env.state = np.clip(np.array([np.cos(ang) * r, np.sin(ang) * r]),
                                0.0, env.size)
        obs = env._observation()
        while True:
            obs, _, terminated, truncated, _ = env.step(act_fn(obs))
            if terminated:
                wins += 1
                break
            if truncated:
                break
    return wins / len(goals)


def make_task(name):
    """返回 (训练环境, 评测环境, 专家, 评测目标列表)。"""
    if name == 'grid':
        goals = np.random.default_rng(10000).uniform([3.5, 3.5], [4.5, 4.5], size=(20, 2))
        return GridWorldEnv(), GridWorldEnv(), Expert(), goals
    return CorridorEnv(), CorridorEnv(), CorridorExpert(seed=0), None


def main():
    p = argparse.ArgumentParser(description='最小行为克隆：示范条数 → 成功率')
    p.add_argument('--task', choices=['grid', 'corridor'], default='grid')
    p.add_argument('--demos', type=int, nargs='+', default=[5, 10, 20, 50, 100])
    p.add_argument('--seeds', type=int, default=3, help='每个示范条数重复几次（训练种子不同）')
    p.add_argument('--demo-noise', type=float, default=0.0,
                   help='示范动作上的高斯噪声；0 表示完美专家')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--perturb', type=float, default=0.0,
                   help='偏移起点评测的半径；0 关闭（只对 grid 任务有意义）')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    train_env, eval_env, expert, goals = make_task(args.task)
    out = args.out or f'results/bc_{args.task}.csv'

    base = evaluate(expert, eval_env, goals)
    print(f'专家上界：成功率={base:.2f}')

    rows = []
    for n_demo in args.demos:
        for s in range(args.seeds):
            states, actions, _ = collect(train_env, expert, n_demo,
                                         seed=1000 + s, action_noise=args.demo_noise)
            policy, mse = train_bc(states, actions, epochs=args.epochs, seed=s)
            act_fn = lambda o, _p=policy: bc_act(_p, o)
            succ = evaluate(act_fn, eval_env, goals, seed=2000 + s)
            succ_shift = (evaluate(act_fn, eval_env, goals,
                                   start_radius=args.perturb, seed=3000 + s)
                          if args.perturb > 0 else 0.0)
            rows.append((n_demo, s, len(states), mse, succ, succ_shift))
            print(f'demos={n_demo:3d} seed={s} 样本={len(states):4d} '
                  f'MSE={mse:.4f} 成功率={succ:.2f} 偏移起点={succ_shift:.2f}')

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('task,demos,seed,n_samples,train_mse,success,success_shifted\n')
        for r in rows:
            f.write(f'{args.task},{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.4f},{r[5]:.4f}\n')
    print(f'结果写入 {out}')


if __name__ == '__main__':
    main()
