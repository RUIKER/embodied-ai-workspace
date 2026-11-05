"""HalfCheetah-v5 上的单次 SAC 训练。

用法：
    python run_one.py --seed 0 --steps 1000000 --out results/raw/sac_s0.json
    python run_one.py --seed 0 --steps 20000 --minimal-size 1000  # smoke

复用 11_SAC/Pendulum 的 SAC 类；本地 networks.py 把 actor 加深到 2×256。

训练协议：
    - 环境步和更新步 1:1（每个 env step 后做一次梯度更新），这是 MuJoCo
      benchmark 的常规做法；Pendulum 那种每个回合更新多次的写法不适用。
    - 前 minimal_size 步只填 buffer，不更新。
    - 训练用随机策略采样动作（SAC 自带熵探索），评估用确定性 tanh(mu)。

输出 JSON 结构：
    train   : 每个回合结束时的全局步数和该回合回报（记录原始 reward，未缩放）
    eval    : 每 eval_interval 步在固定评估种子上的 5 回合平均回报
    final_eval: 训练结束后 20 回合的 mean/median/std/best
    config  : 实际使用的超参（命令行覆盖后的值）
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SAC_DIR = REPO / '11_SAC' / 'Pendulum'
# 先放本地目录，让 agents.py 的 `from networks import ...` 用到本地的 2×256 网络。
sys.path.insert(0, str(SAC_DIR))
sys.path.insert(0, str(HERE))

import settings  # noqa: E402
from agents import SAC  # noqa: E402
from rl_utils import ReplayBuffer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=settings.TOTAL_STEPS)
    parser.add_argument('--eval-interval', type=int, default=settings.EVAL_INTERVAL)
    parser.add_argument('--eval-episodes', type=int, default=settings.EVAL_EPISODES)
    parser.add_argument('--final-eval-episodes', type=int,
                        default=settings.FINAL_EVAL_EPISODES)
    parser.add_argument('--minimal-size', type=int, default=settings.MINIMAL_SIZE)
    parser.add_argument('--device', default=None)
    # 超参覆盖：默认值取自 settings.py，用于 experiments/ablations/sac_hparams 的扫描。
    parser.add_argument('--hidden-dim', type=int, default=settings.HIDDEN_DIM)
    parser.add_argument('--actor-lr', type=float, default=settings.ACTOR_LR)
    parser.add_argument('--critic-lr', type=float, default=settings.CRITIC_LR)
    parser.add_argument('--alpha-lr', type=float, default=settings.ALPHA_LR)
    parser.add_argument('--tau', type=float, default=settings.TAU)
    parser.add_argument('--gamma', type=float, default=settings.GAMMA)
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE)
    parser.add_argument('--reward-scale', type=float, default=settings.REWARD_SCALE)
    parser.add_argument('--init-alpha', type=float, default=None,
                        help='覆盖 SAC 类里写死的初始 alpha=0.01')
    parser.add_argument('--target-entropy', type=float, default=None,
                        help='自动调温的目标熵；默认 -action_dim（HalfCheetah 为 -6）')
    parser.add_argument('--utd', type=int, default=1,
                        help='update-to-data：每个 env step 做几次梯度更新，默认 1')
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def set_seed(seed):
    """固定所有随机源，保证同一 seed 重跑得到同一条曲线。

    cudnn.deterministic=True 会牺牲一点速度换取可复现；
    torch.set_num_threads(1) 是为了多个 seed 并行时不互相抢 CPU 线程。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def deterministic_action(agent, obs):
    """评估用：取策略均值 tanh(mu)，不采样、不加噪。

    训练时动作来自 rsample()，评估时用均值才能反映策略“想做什么”；
    注意不能用 agent.take_action()，那个会继续随机采样。
    """
    state = torch.as_tensor(
        np.asarray(obs), dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    with torch.no_grad():
        hidden = F.relu(agent.actor.fc1(state))
        hidden = F.relu(agent.actor.fc2(hidden))
        mean = agent.actor.fc_mu(hidden)
        action = agent.actor.action_bias + agent.actor.action_scale * torch.tanh(mean)
    return action.squeeze(0).cpu().numpy()


def evaluate(env, agent, num_episodes, seed):
    """在固定初始状态上跑 num_episodes 个完整回合，返回逐回合回报。

    seed 由调用方给定：同一个评估点的初始状态对所有训练 seed 都相同，
    这样不同 seed 之间的差异只来自策略，不来自初始状态。
    """
    returns = []
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            action = deterministic_action(agent, obs)
            if not np.all(np.isfinite(action)):
                # 网络已发散（NaN/Inf）：整次评估记为 NaN，由聚合脚本标成“发散”。
                return np.full(num_episodes, np.nan, dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
        returns.append(episode_return)
    return np.asarray(returns, dtype=np.float32)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else settings.DEVICE

    env = gym.make(settings.ENV_ID)
    eval_env = gym.make(settings.ENV_ID)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low.copy()
    action_high = env.action_space.high.copy()

    agent = SAC(
        state_dim, args.hidden_dim, action_low, action_high,
        args.actor_lr, args.critic_lr, args.alpha_lr,
        args.target_entropy if args.target_entropy is not None else -action_dim,
        args.tau, args.gamma,
        settings.REWARD_SHIFT, args.reward_scale, device,
    )
    if args.init_alpha is not None:
        # log_alpha 与其优化器共享同一个张量，原地改值即可。
        agent.log_alpha.data.fill_(float(np.log(args.init_alpha)))
    buffer = ReplayBuffer(settings.BUFFER_SIZE)

    print(
        f'[sac seed={args.seed}] device={device} env={settings.ENV_ID} '
        f'obs={state_dim} act={action_dim} steps={args.steps}',
        flush=True,
    )
    if device.type == 'cuda':
        print(f'CUDA: {torch.cuda.get_device_name(device)}', flush=True)

    train_steps, train_returns = [], []
    eval_steps, eval_returns = [], []
    global_step = 0
    eval_index = 0
    diverged_at = None
    started = time.perf_counter()

    obs, _ = env.reset(seed=args.seed)
    while global_step < args.steps:
        done = False
        episode_return = 0.0
        while not done and global_step < args.steps:
            try:
                action = agent.take_action(obs)
            except ValueError:
                # 参数变成 NaN 后 torch.distributions.Normal 会直接抛 ValueError。
                action = None
            if action is None or not np.all(np.isfinite(action)):
                diverged_at = global_step
                break
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # 只把 terminated 当作 done 存进 buffer：truncated 是回合时间上限，
            # 不是真正的终止，Bellman target 在 truncated 时仍应 bootstrap。
            # 另外 buffer 里存的是原始 reward；reward_shift/scale 只在
            # SAC.update 计算 critic loss 时施加，这样记录的 return 仍是真实回报。
            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += float(reward)
            global_step += 1

            if buffer.size() >= args.minimal_size:
                # UTD>1 时每个 env step 多次更新，每次重新采样一个 batch。
                try:
                    for _ in range(args.utd):
                        agent.update(buffer.sample(args.batch_size))
                except ValueError:
                    diverged_at = global_step
                    break

            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                # 每个评估点换一组初始状态（+100 + 10*i），
                # 但所有 seed 共用这串 EVAL_SEED，横向比较是同题同卷。
                values = evaluate(
                    eval_env, agent, args.eval_episodes,
                    settings.EVAL_SEED + 100 + eval_index * 10,
                )
                eval_steps.append(int(global_step))
                eval_returns.append(float(values.mean()))
                eval_index += 1

        if diverged_at is not None:
            print(f'[sac seed={args.seed}] DIVERGED at step {diverged_at}', flush=True)
            break
        train_steps.append(int(global_step))
        train_returns.append(float(episode_return))
        if global_step < args.steps:
            # 只有第一个回合用 env.reset(seed=args.seed)，之后不再播种：
            # 每个回合的初始状态应由环境自己随机推进，否则所有回合都从同一点出发，
            # 训练分布会退化成单一轨迹。
            obs, _ = env.reset()

        if global_step % 50_000 < 1_000 or global_step >= args.steps:
            print(
                f'[sac seed={args.seed}] step={global_step} '
                f'episodes={len(train_returns)} '
                f'last10={np.mean(train_returns[-10:]):.1f} '
                f'elapsed={time.perf_counter() - started:.0f}s',
                flush=True,
            )

    wall = time.perf_counter() - started
    if diverged_at is not None and args.eval_interval > 0:
        # 把缺失的评估点补成 NaN，保证所有 run 的评估网格等长、可逐点对齐。
        while len(eval_steps) < args.steps // args.eval_interval:
            eval_steps.append((len(eval_steps) + 1) * args.eval_interval)
            eval_returns.append(float('nan'))
    # 最终评估用 20 回合（比训练中间的 5 回合多）：终点数字要进报告，
    # 4 倍的样本量能把标准误降到约一半。
    final_returns = evaluate(
        eval_env, agent, args.final_eval_episodes, settings.EVAL_SEED
    )

    payload = {
        'algo': 'sac', 'env': settings.ENV_ID, 'seed': args.seed,
        'steps': args.steps, 'device': str(device),
        'config': {
            'hidden_dim': args.hidden_dim,
            'actor_lr': args.actor_lr,
            'critic_lr': args.critic_lr,
            'alpha_lr': args.alpha_lr,
            'gamma': args.gamma, 'tau': args.tau,
            'batch_size': args.batch_size,
            'buffer_size': settings.BUFFER_SIZE,
            'minimal_size': args.minimal_size,
            'reward_shift': settings.REWARD_SHIFT,
            'reward_scale': args.reward_scale,
            'init_alpha': args.init_alpha if args.init_alpha is not None else 0.01,
            'target_entropy': (args.target_entropy if args.target_entropy is not None
                               else -action_dim),
            'utd': args.utd,
        },
        'train': {
            'steps': train_steps,
            'returns': train_returns,
            'wall_clock_sec': wall,
        },
        'eval': {
            'steps': eval_steps,
            'returns': eval_returns,
        },
        'final_eval': {
            'returns': [float(x) for x in final_returns],
            'mean': float(final_returns.mean()),
            'median': float(np.median(final_returns)),
            'std': float(final_returns.std()),
            'best': float(final_returns.max()),
        },
        'wall_clock_sec': float(wall),
        'diverged_at_step': diverged_at,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    print('RESULT', json.dumps({
        'seed': args.seed, 'steps': args.steps,
        'final_mean': payload['final_eval']['mean'],
        'wall': round(wall, 1),
    }), flush=True)
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
