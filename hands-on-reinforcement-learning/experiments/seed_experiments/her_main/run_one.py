"""单次 HER 实验：参数化 HER 开关 / 比例 / 策略 / seed，结果写 JSON，不弹窗。

用法：
    python run_one.py --seed 0 --her 1 --out results/raw/her_seed0.json
    python run_one.py --seed 0 --her 0 --out results/raw/noher_seed0.json

和 main.py 的分工：
- main.py     用 config 里的默认值演示一次、画图看效果（适合交互式学习）；
- 本文件      所有变化量都走命令行，结果只写 JSON（适合批量跑 seed 和消融），
              不 import matplotlib，所以在服务器 / 无显示环境里也能跑。

聚合多个 JSON 的脚本是 aggregate.py。
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GRIDWORLD = REPO / '12_HER' / 'GridWorld'
sys.path.insert(0, str(GRIDWORLD))

import numpy as np
import torch

import config
from agents import DDPG
from environment import make_environment
from rl_utils import ReplayBufferTrajectory
from trainer import evaluate_success, make_eval_goals, train_her_agent


def parse_args():
    """命令行参数。默认值全部取自 config.py，避免两处各写一套超参。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    # 用 int 而不是 store_true：批量脚本里方便写 --her 0 / --her 1。
    parser.add_argument('--her', type=int, default=1, help='1=开启 HER, 0=纯 DDPG')
    parser.add_argument('--ratio', type=float, default=config.HER_RATIO)
    parser.add_argument('--strategy', default=config.GOAL_STRATEGY)
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES)
    # 评估比 main.py 更频繁（50 vs 100），因为批量跑时想更密地看到收敛过程。
    parser.add_argument('--eval-interval', type=int, default=50)
    # --device cpu / cuda；不传就用 config.DEVICE。
    parser.add_argument('--device', default=None)
    # 必填：结果 JSON 的路径，父目录不存在会自动创建。
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def set_seed(seed):
    """固定 Python / NumPy / PyTorch 随机源。与 main.set_seed 相同。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env():
    """按 config 建环境。与 main.make_env 相同。"""
    return make_environment(
        max_steps=config.MAX_STEPS,
        distance_threshold=config.DISTANCE_THRESHOLD,
        action_bound=config.ACTION_BOUND,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    # 命令行显式给了 device 就用它，否则用 config 的自动选择。
    device = torch.device(args.device) if args.device else config.DEVICE

    # 训练 / 评估两个环境，理由见 main.py。
    env, state_dim, action_dim, _, _ = make_env()
    eval_env, _, _, _, _ = make_env()

    # 注意：这里 network 的结构超参全部来自 config，
    # 命令行只暴露"实验设计"层面的变量（seed / HER / 回合数），
    # 这样批量结果之间的差异只来自这些变量，而不是学习率之类的意外改动。
    agent = DDPG(
        state_dim, config.HIDDEN_DIM, action_dim, config.ACTION_BOUND,
        config.ACTOR_LR, config.CRITIC_LR, config.SIGMA, config.TAU,
        config.GAMMA, device,
    )
    buffer = ReplayBufferTrajectory(
        config.BUFFER_SIZE, env, use_her=bool(args.her),
        her_ratio=args.ratio, goal_strategy=args.strategy,
    )
    eval_goals = make_eval_goals(
        config.EVAL_SEED, config.EVAL_GOALS, env.goal_low, env.goal_high
    )

    # 计时用 perf_counter（单调时钟），不受系统时间校准影响。
    started = time.perf_counter()
    returns, history = train_her_agent(
        env, eval_env, agent, buffer, args.episodes,
        config.MINIMAL_EPISODES, config.N_TRAIN, config.BATCH_SIZE,
        args.eval_interval, eval_goals, seed=args.seed,
        use_her=bool(args.her),
    )
    wall = time.perf_counter() - started
    final = evaluate_success(eval_env, agent, eval_goals)

    # JSON 只存聚合脚本真正会用到的字段：
    # - returns / success_history 是原始曲线，aggregate.py 从这里重算指标；
    # - final_success / wall_clock_sec 是训练结束时的快照。
    # 不存网络权重：这里关心的是"HER 有没有用"，不是复现某一次推理。
    payload = {
        'her': bool(args.her), 'ratio': args.ratio,
        'strategy': args.strategy, 'seed': args.seed,
        'episodes': args.episodes, 'device': str(device),
        'returns': [float(r) for r in returns],
        'success_history': [[int(e), float(s)] for e, s in history],
        'final_success': float(final),
        'wall_clock_sec': float(wall),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    # 批量脚本（run_all.sh）靠这一行 RESULT 前缀解析进度；
    # flush=True 保证重定向到日志文件时也能实时看到。
    print('RESULT', json.dumps({
        'her': payload['her'], 'ratio': args.ratio, 'strategy': args.strategy,
        'seed': args.seed, 'final_success': final, 'wall': round(wall, 1),
    }), flush=True)


if __name__ == '__main__':
    main()
