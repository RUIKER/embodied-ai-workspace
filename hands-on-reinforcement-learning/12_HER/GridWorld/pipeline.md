# practice.py 拆解：DDPG + HER 的 pipeline

> 2026-10-28：对应的 pipeline 文件已实现，见 README 的「文件结构」。

对象：practice.py（教程第 19 章原版，254 行）。行号指该文件。
对照：09_DDPG/Pendulum/（本仓库的 DDPG）。

一句话：HER 不是新算法。DDPG 类（L64-132）几乎不动，HER 新增四件事：
环境把目标放进状态、buffer 存整条轨迹、sample() 里换目标并重算奖励、
训练节奏从"每步更新"变成"每条轨迹更新 20 次"。

## 1. 总 pipeline

每个 episode：
  (1) 采样：env.reset() 后，用 actor + 高斯噪声跑完整条轨迹（期间不更新）
      state = [x, y, gx, gy]，每步 traj.store_step(a, s', r, done)
  (2) 入库：replay_buffer.add_trajectory(traj)，存整条，不拆成 transition
  (3) 学习：buffer 轨迹数 >= 200 后，重复 N_TRAIN=20 次：
        batch = buffer.sample(256, use_her=True)
          挑轨迹 -> 挑时刻 t -> 以 80% 概率把目标换成 t 之后到过的位置
          -> 用新目标重算 reward / done，并改写 state 的目标两维
        agent.update(batch)   与普通 DDPG 相同

## 2. 分段对照

| 段落 | 行号 | 作用 | 对应 README 文件 | 相对 DDPG |
| --- | --- | --- | --- | --- |
| WorldEnv | 10-37 | 目标导向环境，奖励依赖目标 | environment.py（已写） | 新 |
| PolicyNet / QValueNet | 39-62 | actor / critic | networks.py | 结构相同，输入维度为 4 |
| DDPG | 64-132 | 智能体 | agents.py | 基本相同（见第 4 节） |
| Trajectory | 134-148 | 记录一条轨迹 | rl_utils.py | 新 |
| ReplayBuffer_Trajectory | 151-195 | 存轨迹 + HER 重标注 | rl_utils.py | 新，核心 |
| 超参数 | 197-212 | | config.py（已写） | N_TRAIN、MINIMAL_EPISODES 的单位变为轨迹 |
| 训练循环 | 214-248 | | trainer.py / main.py | 节奏不同 |
| 画图 | 250-255 | 训练回报曲线 | visualization.py | 只画训练回报，没有评估 |

## 3. 逐段注释

### 3.1 环境（L10-37）

    def reset(self):
        # 目标 g 在 [3.5, 4.5] x [3.5, 4.5] 内随机，起点固定为原点
        self.goal = np.array([4 + random.uniform(-0.5, 0.5), 4 + random.uniform(-0.5, 0.5)])
        self.state = np.array([0, 0])
        self.count = 0
        return np.hstack((self.state, self.goal))      # 观测 = [x, y, gx, gy]

    def step(self, action):
        action = np.clip(action, -1, 1)
        x = max(0, min(5, self.state[0] + action[0]))   # 位置 = 上一位置 + 动作，围墙 [0, 5]
        y = max(0, min(5, self.state[1] + action[1]))
        self.state = np.array([x, y])
        self.count += 1
        dis = ||state - goal||
        reward = -1.0 if dis > 0.15 else 0             # 稀疏奖励：每步 -1，到达为 0
        if dis <= 0.15 or self.count == 50:            # 成功与超时都记为 done
            done = True
        return np.hstack((self.state, self.goal)), reward, done

要点：
1. 观测里带着目标。同一位置换一个目标，网络看到的输入就不同，这是目标导向 RL 的定义。
2. 令 phi(s) = s[:2]，即"实际到达的位置"只由前两维决定。后面所有 [:2] 都是在取 phi。
3. 奖励是 r(s', g) 的函数，给定位置和目标即可重算。HER 能成立依赖这一点。
   Pendulum 的奖励是环境内部的黑盒，无法事后换目标重算。

### 3.2 网络（L39-62）

与 DDPG 同构：actor 输出 tanh(.) * action_bound；critic 拼接 [state, action]，输出标量 Q。
唯一变化是 state_dim = 4。网络本身并不知道"目标"是什么。

### 3.3 DDPG 类（L64-132）

take_action 加高斯噪声，soft_update 用 tau，update 做 critic 的 TD 回归和 actor 的最大化 Q，
与 09_DDPG 一致。与本仓库 DDPG 的小差异见第 4、5 节。

### 3.4 Trajectory（L134-148）

    class Trajectory:
        def __init__(self, init_state):
            self.states = [init_state]    # 比 actions 多 1 个：s0 ... sT
            self.actions = []
            self.rewards = []
            self.dones = []
            self.length = 0               # 等于 len(actions)
        def store_step(self, action, state, reward, done):
            self.actions.append(action)
            self.states.append(state)     # 存执行动作之后的 s_{t+1}
            ...

为什么存整条：future 策略要取"同一条轨迹里 t 之后"的状态。按 transition 打散存，就找不到"之后"。

### 3.5 ReplayBuffer_Trajectory.sample（L162-195）

    for _ in range(batch_size):
        traj = random.sample(self.buffer, 1)[0]          # (1) 随机挑一条轨迹
        step_state = np.random.randint(traj.length)      # (2) 挑时刻 t，t in [0, T-1]
        state      = traj.states[step_state]             #     s_t
        next_state = traj.states[step_state + 1]         #     s_{t+1}
        action     = traj.actions[step_state]
        reward     = traj.rewards[step_state]            #     原目标下的奖励
        done       = traj.dones[step_state]              #     原目标下的 done

        if use_her and np.random.uniform() <= her_ratio: # (3) 80% 概率做 HER
            step_goal = np.random.randint(step_state + 1, traj.length + 1)
            goal = traj.states[step_goal][:2]            # future 策略：假目标 = 之后到过的位置
            dis = ||next_state[:2] - goal||              # (4) 用 s_{t+1} 算，不用 s_t
            reward = -1.0 if dis > dis_threshold else 0  #     奖励按新目标重算
            done   = (dis <= dis_threshold)              #     done 也按新目标重算
            state      = np.hstack((state[:2], goal))    # (5) s 与 s' 的目标两维换成同一个 g
            next_state = np.hstack((next_state[:2], goal))

逐点说明：
- (3) 重标注发生在采样时，不是存储时。同一条数据每次被采到，都可能换成不同的假目标，
  等于放大了样本量。
- (4) 奖励描述的是"执行 a_t 之后是否到达"，所以看 s_{t+1}。用 s_t 是常见 bug。
- (5) Q 学的是 Q(s, a | g)。前后目标不一致，Bellman 方程就不成立。
- 边界情况：step_goal 的下界是 t+1。当 step_goal = t+1 时，假目标恰好等于 s_{t+1}，必然成功。
  对长度为 T 的轨迹，这类"保证成功"样本占重标注样本的比例为 H_T / T（H_T 为调和数）。
  T=50 时约 9%，T=10 时约 29%。我用 T=10、HER_RATIO=1.0 实测 done=True 占 28.6%，与理论吻合。
- 失败轨迹因此被"救"回来：原目标下全是 -1 的轨迹，换成假目标后会出现 0 和 done=True。
  实测：一条超时轨迹的原始 reward 全为 -1；重标注后 reward 取值集合为 {-1, 0}。

### 3.6 训练循环（L214-248）

    for i_episode in range(...):
        state = env.reset()
        traj = Trajectory(state)
        done = False
        while not done:                                    # 先跑完整条轨迹，期间不更新
            action = agent.take_action(state)
            state, reward, done = env.step(action)
            traj.store_step(action, state, reward, done)
        replay_buffer.add_trajectory(traj)                 # 整条入库
        if replay_buffer.size() >= minimal_episodes:       # 单位是轨迹，攒够 200 条才开始
            for _ in range(n_train):                       # 每条轨迹之后做 20 次更新
                transition_dict = replay_buffer.sample(batch_size, True)   # use_her 写死为 True
                agent.update(transition_dict)

## 4. 与 DDPG 的差异

| 维度 | DDPG（09_DDPG/Pendulum） | HER（practice.py） |
| --- | --- | --- |
| 观测 | [cos, sin, 角速度] | [x, y, gx, gy]，目标在观测里 |
| 奖励 | 稠密，环境内部计算 | 稀疏 {-1, 0}，且是 r(s', g)，可事后重算 |
| done | terminated，不把 truncated 当终止 | 成功与超时合并为一个 done |
| buffer 存什么 | 单步 transition | 整条轨迹 Trajectory |
| 采样 | 直接随机取 transition | 先取轨迹、再取时刻、再重标注 |
| 重标注 | 无 | 80% 换目标，重算 reward、done 以及状态的目标两维 |
| 开始学习的条件 | buffer >= 1000 条 transition | buffer >= 200 条轨迹 |
| 更新节奏 | 每个 env step 更新 1 次 | 每条轨迹后更新 20 次 |
| 探索噪声 | 高斯或 OU，加完后 clip | 高斯 sigma=0.1，不 clip（env 内 clip） |
| target 计算 | 在 no_grad 内 | 没有 no_grad |
| 评估 | 训练回报 + 确定性评估 | 只有带噪声的训练回报 |
| 软更新、损失、优化器 | tau、MSE、Adam | 完全相同 |

判断"哪些是 HER 带来的"：
- 是 HER 带来的：轨迹 buffer、采样时重标注、状态含目标、done 需要重算、更新节奏改为每回合 N 次。
- 不是 HER 带来的（是教程写法或环境差异）：动作不 clip、target 不 no_grad、合并 done、没有评估。

HER 与算法无关：把 DDPG 换成 SAC 时，只需改 update，buffer 与重标注不用动（对应 README 消融 4）。

## 5. 已实测的问题

以下四项都经过实际运行验证。

| # | 问题 | 实测 | 影响 | 修法 |
| --- | --- | --- | --- | --- |
| A | 超时被当成终止 | 走不到目标的轨迹 50 步，最后一步 done=True；不做 HER 时采到 done=True 的比例为 2.1%（理论 1/50） | 未重标注的样本里，超时那一步不做 bootstrap；buffer 混用了两种语义 | env 返回 terminated / truncated，buffer 只存 terminated（environment.py 已拆好，需对接 5 元组） |
| B | 动作没有 clip 就存入 buffer | actor 饱和时 take_action 的最大 abs 为 1.313（边界是 1） | 环境实际执行的是 1.0，critic 学的却是 1.313 | take_action 末尾 np.clip 到 [-bound, bound] |
| C | target 网络累积梯度 | update 之后 target_actor、target_critic 的 grad 都不为 None | 数值上无害，但白做反传、白占显存 | q_targets 放进 torch.no_grad() |
| D | use_her 写死，没有评估 | sample(batch_size, True)；env.reset() 无 seed；只画训练回报 | 无法做 USE_HER / HER_RATIO 消融；稀疏奖励下训练回报区分度差 | 读取 config.py 的开关；用固定 20 个目标测成功率 |

另外两点：
- README 写的失败回报是 -100，但按这份代码（MAX_STEPS=50，每步 -1），失败回报是 -50；
  成功于第 k 步的回报是 -(k-1)。
- sample() 是逐样本的 Python 循环，而且对 deque 做随机下标访问是 O(n)。等主流程跑通后再考虑优化。

## 6. 落到 README 的目标文件

| 文件 | 从 practice.py 搬什么 | 需要修改 |
| --- | --- | --- |
| rl_utils.py | Trajectory、ReplayBuffer_Trajectory | 存 terminated；用 env.compute_reward / is_success 重算，不要手写两份 -1/0 逻辑；her_ratio、strategy 从 config 读取 |
| networks.py | PolicyNet、QValueNet | 与 09_DDPG 的命名和 register_buffer 写法对齐 |
| agents.py | DDPG | 修复 B、C |
| trainer.py | 训练循环 | 对接 env.step 的 5 元组；traj 存 terminated；每 EVAL_INTERVAL 评估成功率 |
| main.py | set_seed 与组装 | env.reset(seed=seed+episode)；把 USE_HER 开关传给 sample |
