"""诊断：ratio=1.0 为什么学不会原任务？

经验总结里原先写的解释是“全部换成假目标，真实目标分布的样本被挤掉”。
这只是推测。这个脚本直接测两件事：

(A) 训练时 sample() 出来的目标，有多大比例落在任务目标区域 [3.5,4.5]^2；
    以及已存轨迹里有多大比例曾经走进过这个区域（探索覆盖）。
    -> 如果 ratio=1.0 的训练目标几乎不在任务区域，"挤掉"的说法成立；
       如果探索根本没到过任务区域，那是"没覆盖到"，不是"被挤掉"。

(B) 训练结束后，在一张更宽的目标网格上评估成功率（不只是任务区域）。
    -> 如果策略在近处目标上成功、远处失败，说明它学会了"到达"，
       只是没覆盖到任务区域；如果处处失败，则是没学会。

用法：
    python diagnose_ratio1.py --ratio 1.0 --seed 0
    python diagnose_ratio1.py --ratio 0.8 --seed 0     # 对照
输出：results/diagnostics/ratio{r}_s{seed}.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRIDWORLD = HERE.parents[2] / '12_HER' / 'GridWorld'
sys.path.insert(0, str(GRIDWORLD))

import numpy as np
import torch

import config
from agents import DDPG
from environment import make_environment
from rl_utils import ReplayBufferTrajectory
from trainer import evaluate_success, make_eval_goals, train_her_agent

BUCKET = 100  # 每 100 条轨迹为一个统计窗口


class RecordingBuffer(ReplayBufferTrajectory):
    """在 sample() 之后记录：被采样到的目标有多少落在任务区域内。"""

    def __init__(self, *args, goal_low, goal_high, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_low = np.asarray(goal_low)
        self.goal_high = np.asarray(goal_high)
        self.sampled = {}  # bucket -> [落在任务区域的样本数, 总样本数]

    def sample(self, batch_size, *args, **kwargs):
        batch = super().sample(batch_size, *args, **kwargs)
        goals = batch['states'][:, 2:]
        inside = np.all(
            (goals >= self.goal_low - 1e-6) & (goals <= self.goal_high + 1e-6),
            axis=1,
        )
        bucket = len(self.buffer) // BUCKET
        stat = self.sampled.setdefault(bucket, [0, 0])
        stat[0] += int(inside.sum())
        stat[1] += int(len(inside))
        return batch


def visited_task_region(traj, low, high):
    """这条轨迹是否曾经走进任务目标区域。"""
    states = np.asarray(traj.states)[:, :2]
    return bool(np.any(np.all((states >= low) & (states <= high), axis=1)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES)
    parser.add_argument('--out', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    def make_env():
        return make_environment(
            max_steps=config.MAX_STEPS,
            distance_threshold=config.DISTANCE_THRESHOLD,
            action_bound=config.ACTION_BOUND,
        )

    env, state_dim, action_dim, _, _ = make_env()
    eval_env, _, _, _, _ = make_env()
    agent = DDPG(
        state_dim, config.HIDDEN_DIM, action_dim, config.ACTION_BOUND,
        config.ACTOR_LR, config.CRITIC_LR, config.SIGMA, config.TAU,
        config.GAMMA, config.DEVICE,
    )
    buffer = RecordingBuffer(
        config.BUFFER_SIZE, env, use_her=args.ratio > 0, her_ratio=args.ratio,
        goal_strategy='future', goal_low=env.goal_low, goal_high=env.goal_high,
    )
    eval_goals = make_eval_goals(
        config.EVAL_SEED, config.EVAL_GOALS, env.goal_low, env.goal_high
    )

    started = time.perf_counter()
    returns, history = train_her_agent(
        env, eval_env, agent, buffer, args.episodes,
        config.MINIMAL_EPISODES, config.N_TRAIN, config.BATCH_SIZE,
        50, eval_goals, seed=args.seed, use_her=args.ratio > 0,
    )
    wall = time.perf_counter() - started
    final_task = evaluate_success(eval_env, agent, eval_goals)

    # (A) 训练目标分布：每个窗口里，被采样目标落在任务区域的比例
    sampled = {
        int(b): (c[0] / c[1] if c[1] else None)
        for b, c in sorted(buffer.sampled.items())
    }

    # (A') 探索覆盖：每 200 条轨迹里，曾走进任务区域的轨迹占比
    low, high = env.goal_low, env.goal_high
    coverage = []
    trajs = list(buffer.buffer)
    for start in range(0, len(trajs), 200):
        chunk = trajs[start:start + 200]
        coverage.append({
            'episodes': [start, start + len(chunk)],
            'frac_visit_task_region': float(np.mean(
                [visited_task_region(t, low, high) for t in chunk]
            )),
        })

    # (B) 宽网格评估：不同位置、不同距离的目标
    grid = [(x, y) for x in (1.0, 2.0, 3.0, 4.0) for y in (1.0, 2.0, 3.0, 4.0)]
    grid_success = {}
    for x, y in grid:
        grid_success['%.1f,%.1f' % (x, y)] = evaluate_success(
            eval_env, agent, [np.array([x, y])]
        )

    payload = {
        'ratio': args.ratio, 'seed': args.seed, 'episodes': args.episodes,
        'final_success_task_goals': float(final_task),
        'sampled_goal_in_task_region_by_bucket': sampled,
        'exploration_coverage': coverage,
        'grid_success': grid_success,
        'grid_success_mean': float(np.mean(list(grid_success.values()))),
        'success_history': [[int(e), float(s)] for e, s in history],
        'wall_clock_sec': float(wall),
    }
    out = Path(args.out) if args.out else (
        HERE / 'results' / 'diagnostics' / f'ratio{args.ratio}_s{args.seed}.json'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print('DIAG', json.dumps({
        'ratio': args.ratio, 'seed': args.seed,
        'task_success': final_task,
        'grid_success_mean': payload['grid_success_mean'],
    }), flush=True)


if __name__ == '__main__':
    main()
