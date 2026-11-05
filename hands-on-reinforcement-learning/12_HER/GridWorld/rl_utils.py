"""HER 的核心：按整条轨迹存储，并在采样时做 hindsight relabeling。

======================================================================
数据流：一条轨迹是怎么变成"成功样本"的
======================================================================
    ① 采样（trainer.py）
        episode 跑完 → Trajectory（states / actions / rewards / terminated）
                     → ReplayBufferTrajectory.add_trajectory

    ② 采样训练 batch（本文件 sample）
        随机挑一条轨迹 → 随机挑时刻 t
        ├─ 以 her_ratio 的概率重标注：
        │     从 s_{t+1} … s_T 里取一个位置当假目标 g'
        │     r'、done' 按 g' 重算，s_t / s_{t+1} 的目标两维都换成 g'
        └─ 否则原样返回（真实目标）
        → 拼成 batch 交给 DDPG.update

    ③ 关键点：同一组 (s, a, s') 每次被采样都可能配上不同的 g'。
       原本"全失败、没梯度"的数据，因此被反复利用成大量成功样本，
       这就是 HER 把稀疏奖励任务从"学不会"变成"几百回合学会"的原因。

======================================================================
和 09_DDPG/Pendulum/rl_utils.py 的 ReplayBuffer 的区别
======================================================================
- DDPG 存单步 (s, a, r, s', done)，采样直接随机取 transition；
- HER 必须存整条轨迹，因为 future 策略要取"同一条轨迹里 t 之后"的状态，
  按 transition 存就把时序信息丢了；
- HER 在 sample() 里换目标、重算 reward 和 terminated，而不是在存储时做。
  在采样时做，同一条轨迹每被采一次就能换一个新目标，数据被反复复用；
  如果在存储时做，每条轨迹只能固定一个假目标，样本量不会放大。

这个文件是本次实验里唯一真正新增的逻辑，其余文件基本是 DDPG 的搬运。
"""

import collections
import random

import numpy as np


class Trajectory:
    """一条完整轨迹。

    存储布局（T = 轨迹步数，也就是 length）：

        states     = [s0, s1, ..., sT]            长度 T + 1
        actions    = [a0, a1, ..., a_{T-1}]       长度 T
        rewards    = [r0, r1, ..., r_{T-1}]       长度 T
        terminated = [d0, d1, ..., d_{T-1}]       长度 T

    states 比 actions 多一个，因为 sT 是最后一步执行后的终态——
    HER 的 future 策略可能正好选中它当假目标，所以必须存下来。

    每一步的 transition 就是 (states[t], actions[t], rewards[t],
    states[t+1], terminated[t])。
    """

    def __init__(self, init_state):
        """Args:
            init_state: reset() 返回的观测，作为轨迹的第一个状态 s0。
        """
        # 存拷贝：调用方的 state 数组之后还会被原地改写（env 每次都新建，
        # 但 buffer 不能假设这一点），拷贝最稳。
        self.states = [np.asarray(init_state, dtype=np.float32).copy()]
        self.actions = []
        self.rewards = []
        self.terminated = []
        self.length = 0

    def store_step(self, action, next_state, reward, terminated):
        """存一步。next_state 是执行动作之后的状态 s_{t+1}。

        参数里的 action 必须是环境实际执行的动作（已经 clip 过），
        否则 critic 学到的 (s, a) 和环境真实动力学对不上。

        terminated 只传"到达目标"，不能把"超时"传进来：
        超时与目标无关，它不是 Bellman 意义上的终止，
        而且在 HER 换目标后更不该保留原来的 done。
        """
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.states.append(np.asarray(next_state, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        # 只存 terminated，不存 truncated：超时不是"到达目标"，
        # 不能作为 Bellman target 的终止信号。
        self.terminated.append(bool(terminated))
        self.length += 1

    def __len__(self):
        """让 len(traj) 等于步数，写日志和调试时方便。"""
        return self.length


class ReplayBufferTrajectory:
    """存整条轨迹的 replay buffer，并在 sample() 里做 HER 重标注。

    buffer 的容量单位是**轨迹条数**，不是 transition 条数。
    capacity=10000、每条轨迹最多 50 步，最多约 50 万个 transition。
    """

    def __init__(self, capacity, env, use_her=True, her_ratio=0.8,
                 goal_strategy='future'):
        """Args:
            capacity:      最多存多少条轨迹（deque 满了自动挤掉最旧的）。
            env:           用来调 compute_reward / is_success。
                           重标注的评分函数必须和环境完全一致，所以直接传 env，
                           而不是在 buffer 里另写一份 −1/0 逻辑。
            use_her:       总开关。False 时 sample() 退化成普通轨迹采样，
                           等价于"用轨迹存数据的纯 DDPG"，做消融对照组。
            her_ratio:     每个被采样的 transition 做重标注的概率。
                           注意不是"batch 里恰好 80%"，而是逐条独立抛硬币，
                           所以实际比例在 0.8 附近波动（B=256 时标准差约 0.025）。
            goal_strategy: 'future' / 'final' / 'episode'，见 _choose_goal。
        """
        self.buffer = collections.deque(maxlen=capacity)
        # 重标注时直接用环境的评分函数，避免两份 -1/0 逻辑不一致。
        self.env = env
        self.use_her = bool(use_her)
        self.her_ratio = float(her_ratio)
        self.goal_strategy = goal_strategy
        if goal_strategy not in ('future', 'final', 'episode'):
            raise ValueError(
                "goal_strategy 必须是 'future' / 'final' / 'episode'"
            )

    def add_trajectory(self, trajectory):
        """把一整条跑完的轨迹放进来。只在 episode 结束后调用。"""
        self.buffer.append(trajectory)

    def size(self):
        """当前存了多少条轨迹（而不是多少 transition）。"""
        return len(self.buffer)

    def _choose_goal(self, traj, t, strategy):
        """按策略从轨迹里选一个假目标，只取位置两维作为 φ。

        Args:
            traj:     轨迹。
            t:        当前 transition 的起始时刻（动作 a_t 发生在 s_t）。
            strategy: 目标采样策略。

        Returns:
            形状 (2,) 的假目标 g' = φ(s_k)。
        """
        if strategy == 'future':
            # t 之后（含 t+1 和最后一个状态）等概率选一个时刻。
            # 这是 HER 原论文里效果最好的 future 策略：
            # 目标一定在"未来真正去过的地方"，既可达，又不会像 episode
            # 策略那样频繁选到 s_t 之前已经走过的点（那些点构成平凡目标）。
            # 注意范围包含 t+1：当 t = T−1 时只能选 k = T，
            # 此时 g' = φ(s_T) 恰好就是这一步实际到达的位置，重标注后
            # 这条样本的奖励为 0——这正是"把失败变成成功"的最直接形式。
            k = random.randint(t + 1, traj.length)
        elif strategy == 'final':
            # 只取轨迹最终状态，多样性最差：
            # 所有时刻共用同一个 g'，样本多样性接近普通 replay。
            k = traj.length
        else:
            # 整条轨迹任意时刻，可能选到 t 之前已经走过的位置。
            # 如果 s_t 恰好就在 g' 附近，这条样本会变成"站着不动也算成功"，
            # 教给 critic 错误的动作偏好，所以通常不如 future。
            k = random.randint(0, traj.length)
        return traj.states[k][:2].copy()

    def sample(self, batch_size, use_her=None, her_ratio=None,
               goal_strategy=None):
        """采样一个 batch；use_her=True 时按比例做 hindsight relabeling。

        参数留了 None 默认值，方便消融时临时覆盖 config 里的开关，
        而不用改 buffer 的构造参数。

        流程（对 batch 里每一条独立执行）：
            ① 随机挑一条轨迹，随机挑时刻 t           （均匀，不看回报）
            ② 取出 (s_t, a_t, r_t, s_{t+1}, terminated_t)
            ③ 抛硬币决定是否重标注：
               - 选假目标 g' = φ(s_k)，k 按策略决定
               - 用 φ(s_{t+1}) 和 g' 重算 reward / terminated
               - 把 s_t、s_{t+1} 的目标两维都换成 g'
            ④ 拼进 batch
        """
        if len(self.buffer) == 0:
            raise ValueError("replay buffer 为空，不能采样")
        # 三个 None 参数：调用方没显式传就用构造时的配置。
        if use_her is None:
            use_her = self.use_her
        if her_ratio is None:
            her_ratio = self.her_ratio
        if goal_strategy is None:
            goal_strategy = self.goal_strategy

        batch = {
            'states': [], 'actions': [], 'next_states': [],
            'rewards': [], 'dones': [],
        }
        # 用 while 而不是 for：长度为零的轨迹（理论上不会出现，因为
        # 每回合至少走一步）会被跳过，collected 不增加，循环继续。
        collected = 0
        while collected < batch_size:
            # ① 均匀挑轨迹、均匀挑时刻。注意这是"按轨迹均匀"而不是
            #    "按 transition 均匀"：短轨迹里的每一步被采到的概率更高。
            #    对 HER 来说这通常无伤大雅，但值得知道。
            traj = self.buffer[random.randrange(len(self.buffer))]
            if traj.length == 0:
                continue

            t = random.randrange(traj.length)

            # ② 取出一条原始 transition。
            state = traj.states[t]
            next_state = traj.states[t + 1]
            action = traj.actions[t]
            reward = traj.rewards[t]
            terminated = traj.terminated[t]

            # ③ 以 her_ratio 的概率重标注。
            #    逐条独立判断，所以一个 batch 里重标注样本的数量是随机的。
            if use_her and random.random() < her_ratio:
                goal = self._choose_goal(traj, t, goal_strategy)
                # 奖励描述的是"执行 a_t 之后有没有到达"，所以看 s_{t+1}，
                # 不是 s_t。用 s_t 会奖励"还没到就算到"的样本，
                # 这是复现 HER 时最常踩的坑之一（README 第 3 点）。
                achieved = next_state[:2]
                reward = float(self.env.compute_reward(achieved, goal))
                terminated = bool(self.env.is_success(achieved, goal))
                # s_t 与 s_{t+1} 的目标两维必须换成同一个 g'，
                # 否则 Q(s, a | g') 的 Bellman 等式不成立：
                # 网络会看到"起点目标 A、终点目标 B"的样本，两边对不上。
                state = np.concatenate([state[:2], goal])
                next_state = np.concatenate([next_state[:2], goal])
                # action 不做任何修改：动作本身没变，变的是"当时想去哪"。
                #
                # 还有一点：truncated 不在轨迹里（见 Trajectory.store_step），
                # 所以重标注后 done 只可能来自"新目标是否到达"，
                # 不会把"超时"误当成到达，也不会漏掉超时样本的自举。

            batch['states'].append(state)
            batch['actions'].append(action)
            batch['next_states'].append(next_state)
            batch['rewards'].append(reward)
            batch['dones'].append(terminated)
            collected += 1

        # 统一转成 float32 的 ndarray：和 torch 默认 dtype 对齐，
        # 避免每次 update 时再做隐式转换。
        return {
            'states': np.asarray(batch['states'], dtype=np.float32),
            'actions': np.asarray(batch['actions'], dtype=np.float32),
            'next_states': np.asarray(batch['next_states'], dtype=np.float32),
            'rewards': np.asarray(batch['rewards'], dtype=np.float32),
            'dones': np.asarray(batch['dones'], dtype=np.float32),
        }


def moving_average(values, window_size=9):
    """因果滑动平均，输出长度与输入一致（开头窗口逐步变大）。

    对第 i 个元素，取 max(0, i−window+1) 到 i 的平均。
    只往后看、不往前看，避免曲线"提前"表现出未来才有的趋势。

    实现上用前缀和（cumsum）把每次计算降到 O(1)，
    整条曲线 O(n)，比循环调用 np.mean 快一个数量级。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.copy()
    # 在开头补一个 0，这样 cumulative[i] 就等于前 i 个元素之和（1-based）。
    cumulative = np.cumsum(np.insert(values, 0, 0))
    indices = np.arange(1, values.size + 1)
    # 窗口左端点不能早于 0；counts 是当前窗口里的元素个数（开头不足 window）。
    starts = np.maximum(0, indices - window_size)
    counts = indices - starts
    return ((cumulative[indices] - cumulative[starts]) / counts).astype(
        np.float32
    )
