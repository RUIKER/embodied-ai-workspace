"""DDPG + HER 在二维网格世界上的入口。

运行：
    python main.py

消融只改 config.py：
    USE_HER = False        纯 DDPG 对照组
    HER_RATIO = 0.5        调整假目标比例
    GOAL_STRATEGY = 'final' / 'episode'

======================================================================
这个文件只做"装配"，不实现任何算法
======================================================================
    set_seed         固定三条随机流（Python / NumPy / PyTorch）
    make_env         建训练环境 env 和评估环境 eval_env
    DDPG             四个网络 + 两个优化器
    ReplayBuffer...  带 HER 的轨迹 buffer
    train_her_agent  训练 + 定期评估
    最后              打印汇总、画两张图

想看懂 HER，去 rl_utils.py；想看懂 DDPG，去 agents.py。
"""

import random

import numpy as np
import torch

import config
from agents import DDPG
from environment import make_environment
from rl_utils import ReplayBufferTrajectory
from trainer import evaluate_success, make_eval_goals, train_her_agent
from visualization import plot_training_results


def set_seed(seed):
    """固定 Python / NumPy / PyTorch 的随机源。

    需要固定三处，是因为它们各自维护独立的状态：
    - random：buffer 里选轨迹、选时刻、选假目标时刻（rl_utils）；
    - np.random：探索噪声（agents.take_action）和评估目标（部分脚本）；
    - torch：网络参数初始化、dropout 等。
    环境的随机流不走这里，它由 env.reset(seed=...) 单独管理（见 environment.py）。

    deterministic / benchmark 关掉的是 cuDNN 的自动调优，
    让 GPU 卷积结果可复现。这个任务全是全连接层，实际影响接近 0，
    但保留这两行是个好习惯。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env():
    """按 config 建一个 GridWorldEnv。

    返回 5 元组 (env, state_dim, action_dim, action_low, action_high)，
    后三个本来是为对齐 Gymnasium 风格的动作缩放准备的；
    这个任务的动作空间是对称的 [-1, 1]，所以 main 里用不到 low/high。
    """
    return make_environment(
        max_steps=config.MAX_STEPS,
        distance_threshold=config.DISTANCE_THRESHOLD,
        action_bound=config.ACTION_BOUND,
    )


def main():
    # ① 固定随机源，保证同一 seed 下结果可复现。
    set_seed(config.SEED)

    # ② 两个环境实例：
    #    训练环境的 goal 每回合随机；
    #    评估环境只用来 test_set_goal，绝不能在同一个实例上评估，
    #    否则会重置掉正在采样的 episode，轨迹数据就被污染了。
    env, state_dim, action_dim, _, _ = make_env()
    # 评估用独立环境，避免 test_set_goal 打断训练中的 episode。
    eval_env, _, _, _, _ = make_env()

    # ③ agent：状态 4 维（位置 + 目标）、动作 2 维。
    agent = DDPG(
        state_dim=state_dim,
        hidden_dim=config.HIDDEN_DIM,
        action_dim=action_dim,
        action_bound=config.ACTION_BOUND,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.CRITIC_LR,
        sigma=config.SIGMA,
        tau=config.TAU,
        gamma=config.GAMMA,
        device=config.DEVICE,
    )

    # ④ buffer：必须把 env 传进去，采样时要用同一个 compute_reward 重算奖励。
    #    use_her / her_ratio / goal_strategy 三个开关都在这里注入，
    #    后续训练循环不再感知 HER 的存在。
    replay_buffer = ReplayBufferTrajectory(
        capacity=config.BUFFER_SIZE,
        env=env,
        use_her=config.USE_HER,
        her_ratio=config.HER_RATIO,
        goal_strategy=config.GOAL_STRATEGY,
    )

    # ⑤ 评估目标：固定 seed 生成，和训练环境的随机目标无关。
    #    所有基线、所有 seed 都跑这同一组，成功率才能横向比较。
    eval_goals = make_eval_goals(
        config.EVAL_SEED, config.EVAL_GOALS,
        env.goal_low, env.goal_high,
    )

    # ⑥ 训练。use_her 再传一次是让 trainer 在采样时显式覆盖 buffer 配置；
    #    这样消融脚本可以在不改 buffer 构造参数的情况下切换 HER。
    return_list, success_history = train_her_agent(
        env=env,
        eval_env=eval_env,
        agent=agent,
        replay_buffer=replay_buffer,
        num_episodes=config.NUM_EPISODES,
        minimal_episodes=config.MINIMAL_EPISODES,
        n_train=config.N_TRAIN,
        batch_size=config.BATCH_SIZE,
        eval_interval=config.EVAL_INTERVAL,
        eval_goals=eval_goals,
        seed=config.SEED,
        use_her=config.USE_HER,
    )

    # ⑦ 训练结束后再用确定性策略评一次，作为最终指标。
    final_success = evaluate_success(eval_env, agent, eval_goals)
    print('USE_HER=%s  HER_RATIO=%.2f  STRATEGY=%s' % (
        config.USE_HER, config.HER_RATIO, config.GOAL_STRATEGY
    ))
    # 训练回报的最后 10 回合平均值：HER 正常时应该明显高于 −50。
    print('最后 10 回合训练平均回报: %.3f' % np.mean(return_list[-10:]))
    # 成功率才是主指标：固定 20 个目标，到达比例。
    print('最终固定目标成功率: %.3f (%d goals)' % (
        final_success, config.EVAL_GOALS
    ))

    # ⑧ 左图回报曲线（含滑动平均），右图固定目标成功率曲线。
    plot_training_results(
        return_list, success_history,
        env_name='GridWorld', use_her=config.USE_HER,
        num_eval_goals=config.EVAL_GOALS,
    )


if __name__ == '__main__':
    main()
