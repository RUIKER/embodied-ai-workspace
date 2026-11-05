"""HER 的训练循环与评估。

======================================================================
和 09_DDPG/Pendulum/trainer.py 的关键区别
======================================================================
1. 训练数据按"整条轨迹"收集和入 buffer，不是每个 env step 存一条 transition；
   因为 HER 的 future 策略需要在同一条轨迹里往前找假目标。
2. 每个 episode 结束后做 n_train 次更新，而不是每个 env step 更新一次。
3. 评估指标是成功率（固定一组目标），因为稀疏奖励下回报几乎只有 −50 / 0 两档。

======================================================================
一个 episode 的完整流程
======================================================================
    reset(seed 只在第一回合给)
    └─ while not done:
           a = π(s) + 噪声
           s', r, terminated, truncated = env.step(a)
           traj.store_step(a, s', r, terminated)     ← 只存 terminated
    轨迹结束
    ├─ buffer.add_trajectory(traj)
    ├─ 若 buffer 里已有 ≥ minimal_episodes 条轨迹：
    │      重复 n_train 次：batch = buffer.sample(B) ; agent.update(batch)
    └─ 每 eval_interval 回合：在固定目标上跑确定性策略，记成功率

也就是说，warmup 期间（前 200 回合）只采数据不学习；
之后每采 1 条轨迹做 20 次梯度更新。
"""

import numpy as np
from tqdm import tqdm

from rl_utils import Trajectory


def make_eval_goals(seed, num_goals, goal_low, goal_high):
    """生成固定评估集；所有算法、seed、评估点共用同一组目标。

    用独立的 rng，不碰环境的随机流，也不碰全局 np.random。
    seed 固定为 config.EVAL_SEED，所以不同训练 seed 之间评估集完全相同，
    成功率才有可比性。

    Returns:
        list，长度 num_goals，每个元素是形状 (2,) 的目标坐标。
    """
    rng = np.random.default_rng(seed)
    return [rng.uniform(goal_low, goal_high) for _ in range(num_goals)]


def evaluate_success(env, agent, goals):
    """在固定目标上跑确定性策略，返回成功率。

    - 每个目标都从原点重新开始（test_set_goal 会 reset 状态）；
    - explore=False：不加噪声，评估的是策略本身，不是"会探索的策略"；
    - 成功只看 terminated（真正到达目标），不看 truncated；
    - 必须传**独立**的 eval_env，否则会把训练中正在跑的 episode 打断。

    Returns:
        [0, 1] 之间的 float，= 成功目标数 / 目标总数。
    """
    successes = []
    for goal in goals:
        state, _ = env.test_set_goal(goal)
        done = False
        success = False
        # 每回合最多 max_steps 步，最多 20 × 50 = 1000 次前向，
        # 频率控制在每 50~100 回合一次，开销可以忽略。
        while not done:
            action = agent.take_action(state, explore=False)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated:
                success = True
            done = terminated or truncated
        successes.append(1.0 if success else 0.0)
    return float(np.mean(successes))


def train_her_agent(env, eval_env, agent, replay_buffer, num_episodes,
                    minimal_episodes, n_train, batch_size, eval_interval,
                    eval_goals, seed=None, use_her=None):
    """DDPG + HER 的主循环。

    Args:
        env:              训练环境（目标随机）。
        eval_env:         评估环境（目标由 test_set_goal 指定），必须与 env 分开。
        agent:            DDPG。
        replay_buffer:    ReplayBufferTrajectory。
        num_episodes:     训练总回合数。
        minimal_episodes: warmup 门槛，单位是**轨迹条数**，不是 transition 条数。
        n_train:          每采一条轨迹后做多少次梯度更新。
        batch_size:       每次更新的 batch 大小。
        eval_interval:    每多少回合评估一次；<= 0 表示不评估。
        eval_goals:       固定评估目标列表。
        seed:             训练 seed；只在第一回合传给 env.reset。
        use_her:          传给 buffer.sample()，默认 None = 用 buffer 自己的配置。

    参数单位提醒：
    - minimal_episodes 是"轨迹条数"，不是 transition 条数；
    - n_train 是每条轨迹之后做多少次梯度更新。

    Returns:
        return_list:    每回合的训练回报（长度 num_episodes）。
        success_history: [(episode, success_rate), ...]，只含评估点。
    """
    return_list = []
    success_history = []

    with tqdm(total=num_episodes, desc='Training') as progress:
        for episode in range(num_episodes):
            # 只在第一个 episode 播种，之后沿用同一条随机流。
            # 若每回合都 reset(seed=seed+episode)，不同 seed 的目标序列
            # 会变成彼此错位的同一条序列（实测重合比例 100%），seed 不独立。
            if seed is not None and episode == 0:
                state, _ = env.reset(seed=seed)
            else:
                state, _ = env.reset()

            traj = Trajectory(state)
            done = False
            episode_return = 0.0

            # ① 先完整跑完一条轨迹，期间不更新。
            #    这是 HER 的硬性要求：future 策略要等到 s_T 才知道有哪些
            #    未来状态可以当假目标，所以必须先收完再入 buffer。
            while not done:
                action = agent.take_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # 只把 terminated 写入轨迹；truncated 不参与 Bellman 终止，
                # 但它是上面的 done，用来结束采样循环。
                traj.store_step(action, next_state, reward, terminated)
                state = next_state
                episode_return += reward

            # ② 整条轨迹入 buffer，一条轨迹算一个条目。
            replay_buffer.add_trajectory(traj)
            return_list.append(episode_return)

            # ③ 攒够轨迹后，每条轨迹做 n_train 次更新。
            #    这里比较的是"轨迹条数"，不是 transition 条数；
            #    warmup 期间 agent 一次都不更新，策略一直是随机初始化附近的样子，
            #    所以前 200 回合的回报基本是噪声，看曲线时不要误判成"没学会"。
            if replay_buffer.size() >= minimal_episodes:
                for _ in range(n_train):
                    batch = replay_buffer.sample(
                        batch_size, use_her=use_her
                    )
                    agent.update(batch)

            # ④ 定期在固定目标上评估成功率。
            #    第 0 回合（即第 1 回合结束时）也评估一次，作为"随机策略基线"。
            if eval_interval > 0 and (
                episode == 0 or (episode + 1) % eval_interval == 0
            ):
                success = evaluate_success(eval_env, agent, eval_goals)
                success_history.append((episode + 1, success))
                progress.set_postfix({
                    'episode': episode + 1,
                    'success': '%.2f' % success,
                    'return': '%.2f' % np.mean(return_list[-10:]),
                })
            elif (episode + 1) % 10 == 0:
                # 非评估回合也每 10 回合刷新一次进度条，方便肉眼判断是否在学。
                progress.set_postfix({
                    'episode': episode + 1,
                    'return': '%.2f' % np.mean(return_list[-10:]),
                })
            progress.update(1)

    return return_list, success_history
