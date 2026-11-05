"""二维网格世界：一个目标导向（goal-conditioned）的最小环境。

这个环境是 HER 的教学载体：
- 足够简单：状态 4 维、动作 2 维、动力学就是纯平移，读一遍就能在脑子里跑；
- 又足够难：奖励稀疏到随机策略几乎不可能碰巧成功，HER 有无的差别一眼可见。

======================================================================
一、任务定义
======================================================================
5×5 的平面，智能体从原点 (0, 0) 出发，每回合要去一个随机目标点 g。
目标在右上角 [3.5, 4.5] × [3.5, 4.5] 的 1×1 方块里均匀采样，
离原点的距离在 √(3.5²+3.5²) ≈ 4.95 到 √(4.5²+4.5²) ≈ 6.36 之间。

    状态   s  = [x, y, goal_x, goal_y]      4 维
    动作   a  = [dx, dy]                    2 维连续，每维 ∈ [-1, 1]
    动力学 s' = clip(s + a, 0, 5)           纯平移：无惯性、无随机性

到达判定和奖励都只看"位置"和"目标"之间的距离：

    d(s, g)    = ‖φ(s) − g‖₂                φ(s) = s[:2] = 当前位置
    r(s, a, g) =  0    if d(s', g) ≤ 0.15   到达目标
                 −1    其他                  每一步都扣 1 分

======================================================================
二、为什么奖励稀疏到需要 HER
======================================================================
每步最多移动 √2 ≈ 1.41；从原点进目标区最快也要 4 步。
而成功区域只是以目标为圆心、半径 0.15 的圆，面积约 0.071，
只占整个 5×5 平面的 0.28%。

一个不看目标的策略，在 50 步内恰好停进这个圆的概率可以忽略不计。
于是训练早期几乎每条轨迹都是"回报 −50、从未成功"，
critic 的 TD 目标里没有任何"做对了"的信号——这就是稀疏奖励的困境。
HER 要解决的是：一条全失败的轨迹里，其实也记下了"我实际走到了哪里"，
这种信息可以拿来构造"以失败点为目标"的成功样本。

======================================================================
三、φ（achieved goal）为什么只取位置两维
======================================================================
状态里已经写了目标，但有两件不同的事必须分开：

    desired_goal = g                当前任务"要求"去哪（外部给定的）
    achieved_goal = φ(s) = s[:2]    这个状态"实际"在哪（由轨迹决定的）

HER 重标注只替换 desired_goal：把"要求去的地方"换成"实际到过的地方"，
失败轨迹就变成了一条以失败点为目标的成功轨迹。
φ 是整套操作的地基——环境算奖励、buffer 做 relabel、评测判成功，
三处必须用同一个 φ，否则同一条数据会算出互相矛盾的奖励。

======================================================================
四、terminated / truncated 为什么必须分开
======================================================================
    路线                 含义                         HER 重标注时
    terminated=True      到达目标（与目标有关）       要按新目标重算
    truncated=True       超过 max_steps（与目标无关）  不能改，也不参与 Bellman 终止

如果像很多教程那样合成一个 done，relabel 时就分不清"到达"还是"超时"：
把超时当成到达，会教 critic "走到哪都算成功"；
把到达当成超时，会教它"到了也还得继续走"。
两种情况 Bellman 等式两端说的都不是同一件事，训练会悄悄跑偏。

接口按 Gymnasium 的 5 元组写：(obs, reward, terminated, truncated, info)，
以后要换成本项目里的其它环境时，调用方不用改。

======================================================================
五、时间尺度与数值范围（读日志时用得上）
======================================================================
- episode 上限 50 步；
- 成功回报 ≈ −(到达步数)，典型值 −4 ~ −6；失败回报恒为 −50；
- 所以训练回报的滑动平均只在 −50 到 −5 之间跳，
  没有中间地带——这也是评估要看成功率而不是回报的原因（见 trainer.py）。
"""

import numpy as np


class GridWorldEnv:
    """二维平面上从原点走到目标点的任务。

    刻意不继承 gym.Env：整个文件一百多行，每一行都能读懂，
    比"继承一个基类、再覆盖三个方法"更适合当教学材料。
    外部只需要这四个接口：

        env.reset(seed=None)              -> (obs, info)
        env.step(action)                  -> (obs, r, terminated, truncated, info)
        env.compute_reward(φ(s'), g)      -> float   # HER 重标注也调它
        env.test_set_goal(goal)           -> (obs, info)  # 评测用，目标不随机
    """

    def __init__(self,
                 size=5.0,
                 max_steps=50,
                 distance_threshold=0.15,
                 action_bound=1.0,
                 goal_low=(3.5, 3.5),
                 goal_high=(4.5, 4.5)):
        """Args:
            size:               平面边长，位置坐标限制在 [0, size]²。
            max_steps:          单回合最大步数，达到就 truncated。
            distance_threshold: 到达半径；调小任务更难（成功圆面积 ∝ r²）。
            action_bound:       动作每一维的上下限；agent 侧也拿它做 clip。
            goal_low / goal_high: 目标采样矩形的左下角 / 右上角。
        """
        self.size = size
        self.max_steps = max_steps
        self.distance_threshold = distance_threshold
        self.action_bound = action_bound
        self.goal_low = np.asarray(goal_low, dtype=np.float64)
        self.goal_high = np.asarray(goal_high, dtype=np.float64)

        # 维度信息供 main.py 构造网络。
        # observation_dim 是 4 而不是 2：网络必须同时看到"我在哪"和"要去哪"，
        # 否则同一个位置在离目标远和近时对应两种完全不同的最优动作。
        self.observation_dim = 4
        self.action_dim = 2
        self.action_low = -action_bound * np.ones(self.action_dim)
        self.action_high = action_bound * np.ones(self.action_dim)

        # 用独立的 Generator，避免和 numpy 全局随机状态互相污染
        # （多 seed 并行跑实验时，全局 np.random 会被各个 env 互相覆盖）。
        # reset(seed=None) 时沿用上一条随机流，和 Gymnasium 的约定一致。
        self.rng = np.random.default_rng()
        self.state = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.count = 0          # 本回合已走的步数，用来判 truncated

    # ------------------------------------------------------------------ 目标
    @property
    def achieved_goal(self):
        """φ(s)：从状态里提取"实际到达的目标"。

        在这个环境里就是当前位置。换到机械臂、推方块等任务时，
        φ 会复杂得多（末端位姿、方块坐标），但接口保持不变。
        """
        return self.state.copy()

    @property
    def desired_goal(self):
        """当前任务的目标 g。

        HER 训练时它会在 batch 里被替换成假目标（buffer 内部操作），
        环境本身的 goal 不变；评测时用 test_set_goal 固定它。
        """
        return self.goal.copy()

    def _observation(self):
        """拼出网络看到的 4 维观测 [x, y, goal_x, goal_y]。

        注意这里返回的数组是新建的，轨迹里存的是它的一份拷贝，
        所以之后 self.state 变化不会改到已经存进 buffer 的数据。
        """
        return np.concatenate([self.state, self.goal])

    def compute_reward(self, achieved_goal, desired_goal):
        """按 (实际到达, 目标) 这一对算奖励：在阈值内 0，否则 −1。

        HER 重标注时直接调这个，不要另写一份逻辑——
        两份实现迟早会不一致（阈值、范数写法、浮点边界），
        而且这个函数本身就是"评分函数"，重标注和原始奖励必须是同一个。

        参数形状：这里是**单对** (2,) 向量，不支持 (N, 2) 批量。
        如果传入批量，np.linalg.norm 会算成整块矩阵的 Frobenius 范数，
        结果是一个标量而不是 N 个奖励——需要批量时自己沿 axis=1 算。
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if distance <= self.distance_threshold else -1.0

    def is_success(self, achieved_goal, desired_goal):
        """到达判定。和 compute_reward 用同一个距离与阈值。

        单独抽出来，是因为"奖励为 0"和"任务成功"在别的环境里不一定等价
        （比如有中间奖励、或者成功条件不是距离），到那时改这里就行。
        返回 Python bool，方便直接写进轨迹和 JSON。
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return bool(distance <= self.distance_threshold)

    # ---------------------------------------------------------------- 交互
    def reset(self, seed=None, options=None):
        """开始新回合：随机采一个目标，状态回到原点。

        seed 的语义与 Gymnasium 一致：
        - reset(seed=k)：用 k 重建随机流，之后连续的 reset() 沿用这条流；
        - reset()：不重置，继续用当前随机流。
        这样同一 seed 下的目标序列可复现，多 seed 实验之间又互不干扰。

        options 只是为了对齐 Gymnasium 的签名，这里没有可配置项。

        Returns:
            (obs, info)：obs 是 (4,) 的 [x, y, goal_x, goal_y]；
            info 为空 dict（Gymnasium API 要求）。
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 目标固定在右上区域：[3.5, 4.5] × [3.5, 4.5]
        self.goal = self.rng.uniform(self.goal_low, self.goal_high)
        self.state = np.zeros(2, dtype=np.float64)
        self.count = 0
        return self._observation(), {}

    def step(self, action):
        """执行一步，返回 Gymnasium 5 元组。

        奖励用的是**执行之后**的 self.state（即 s'），
        因为奖励描述的是"做了这个动作之后有没有到达"。
        HER 重标注时也必须用 s' 来算，用 s_t 是这套代码最常见的 bug。

        Returns:
            obs, reward, terminated, truncated, info
            - terminated=True：到达目标（HER 会按新目标重算）
            - truncated=True ：步数达到 max_steps（HER 不碰）
        """
        # ① 动作先裁到环境允许的范围。环境自己裁一次是最后一道保险；
        #    agent 侧也必须裁（见 agents.take_action），否则 buffer 里存的
        #    动作和环境真正执行的动作不一致，critic 会学到错误的一对 (s, a)。
        action = np.clip(np.asarray(action, dtype=np.float64),
                         -self.action_bound, self.action_bound)

        # ② 纯平移 + 把位置限制在平面内。撞墙时对应那一维被截住，
        #    另一维照常生效（斜着撞墙可以沿墙滑）——这就是"无随机性"的动力学，
        #    同样的 s 和 a 永远得到同样的 s'。
        self.state = np.clip(self.state + action, 0.0, self.size)
        self.count += 1

        # ③ 先算奖励和终止，再返回。顺序不能反：两者都依赖新的 self.state。
        reward = self.compute_reward(self.state, self.goal)
        terminated = self.is_success(self.state, self.goal)
        # 到达与超时可能同时发生（最后一步刚好到达），此时两个都是 True；
        # 训练侧用 "terminated or truncated" 结束采样，用 terminated 做 Bellman 掩码，
        # 所以这种边界情况不会互相干扰。
        truncated = self.count >= self.max_steps
        return self._observation(), reward, terminated, truncated, {}

    def test_set_goal(self, goal):
        """评测用：把目标设成指定的一个，方便固定评估集。

        和 reset 的区别：目标不随机、也不动随机流。
        评测时所有算法、所有 seed 都跑这同一批 (起点=原点, 目标)，
        成功率才有可比性。评测必须在独立的环境实例上做
        （见 main.py 的 eval_env），否则会打断训练中的 episode。
        """
        self.goal = np.asarray(goal, dtype=np.float64).copy()
        self.state = np.zeros(2, dtype=np.float64)
        self.count = 0
        return self._observation(), {}


def make_environment(**kwargs):
    """建环境，并把构造网络要用的维度/动作范围一起返回。

    返回 5 元组而不是只返回 env，是为了让 main.py / experiments/seed_experiments/her_main/run_one.py
    不必写死 STATE_DIM=4、ACTION_DIM=2；换环境时调用方只改这一行。
    """
    env = GridWorldEnv(**kwargs)
    return env, env.observation_dim, env.action_dim, env.action_low, env.action_high
