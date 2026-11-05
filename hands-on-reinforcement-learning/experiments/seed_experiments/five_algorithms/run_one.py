"""单次训练驱动：DDPG / TD3 / SAC on Pendulum-v1。

设计原则：
1. 不改动三个算法目录里的任何代码，只在运行时按各自 main.py 的方式组装 agent；
2. 训练循环与各自的 trainer.py 保持一致（episode reset 用 seed+episode，
   buffer 里存 terminated 而不是 truncated，每个 env step 更新一次）；
3. 额外记录累计 env steps、periodic 确定性评估：每个评估点都用同一组 20 个初始状态
   （EVAL_SEED..EVAL_SEED+19），所有算法、所有 seed、所有评估点共用；
4. 训练结束后再用同一组初始状态做最终评估。

用法（由 run_compare.py 调用）：
    python experiments/seed_experiments/five_algorithms/run_one.py --algo td3 --seed 0 --episodes 200 \
        --out experiments/seed_experiments/five_algorithms/results/raw/td3_seed0.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ALGO_DIRS = {
    "ddpg": REPO / "09_DDPG" / "Pendulum",
    "td3": REPO / "10_TD3" / "Pendulum",
    "sac": REPO / "11_SAC" / "Pendulum",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one fixed-seed RL experiment.")
    parser.add_argument("--algo", required=True, choices=sorted(ALGO_DIRS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=2000,
                        help="每多少个 env steps 做一次 periodic eval；0 表示关闭")
    parser.add_argument("--device", default="cpu",
                        help="网络很小，默认 CPU；有 CUDA 时可传 cuda")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="覆盖 config 里的超参，例如 --set NOISE_SIGMA=0.1")
    parser.add_argument("--out", required=True,
                        help="单次运行的 JSON 输出路径")
    parser.add_argument("--model-out", default=None,
                        help="可选：保存 actor state_dict 的路径")
    return parser.parse_args()


def set_seed(seed):
    """固定 Python / NumPy / PyTorch（含 CUDA）的随机源。"""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_overrides(config, items):
    for item in items:
        key, value = item.split("=", 1)
        old = getattr(config, key)
        setattr(config, key, type(old)(value))


def build_agent(algo, config, agents, state_dim, action_dim,
                action_low, action_high, device):
    if algo == "ddpg":
        return agents.DDPG(
            state_dim, config.HIDDEN_DIM, action_dim, action_low, action_high,
            config.ACTOR_LR, config.CRITIC_LR, config.GAMMA, config.TAU,
            config.NOISE_TYPE, config.NOISE_SIGMA, config.OU_THETA, device,
        )
    if algo == "td3":
        return agents.TD3(
            state_dim=state_dim,
            hidden_dim=config.HIDDEN_DIM,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            actor_lr=config.ACTOR_LR,
            critic_lr=config.CRITIC_LR,
            gamma=config.GAMMA,
            tau=config.TAU,
            policy_delay=config.POLICY_DELAY,
            target_noise=config.TARGET_NOISE,
            noise_clip=config.NOISE_CLIP,
            exploration_sigma=config.EXPLORATION_SIGMA,
            device=device,
        )
    return agents.SAC(
        state_dim, config.HIDDEN_DIM, action_low, action_high,
        config.ACTOR_LR, config.CRITIC_LR, config.ALPHA_LR,
        -action_dim, config.TAU, config.GAMMA,
        config.REWARD_SHIFT, config.REWARD_SCALE, device,
    )


def deterministic_action(agent, state, algo):
    """评估用动作：关闭探索噪声。

    - DDPG / TD3：actor(state)，本来就是确定性的；
    - SAC：取 tanh(mu) 再做动作缩放，不使用 rsample()/随机采样。
    """
    if algo != "sac":
        return agent.take_action(state, explore=False)

    import torch

    state_t = torch.as_tensor(
        np.asarray(state), dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    with torch.no_grad():
        hidden = torch.relu(agent.actor.fc1(state_t))
        mean = agent.actor.fc_mu(hidden)
        action = agent.actor.action_bias + agent.actor.action_scale * torch.tanh(mean)
    return action.squeeze(0).cpu().numpy()


def evaluate(env, agent, algo, num_episodes, seed):
    """确定性评估；所有算法使用同一组 eval seed，保证初始状态一致。"""
    returns = []
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            action = deterministic_action(agent, state, algo)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(float(episode_return))
    return np.asarray(returns, dtype=np.float32)


def config_snapshot(config):
    """只保留可 JSON 序列化的超参。"""
    snapshot = {}
    for key, value in vars(config).items():
        if key.isupper() and isinstance(value, (int, float, str, bool)):
            snapshot[key] = value
    return snapshot


def main():
    args = parse_args()
    algo_dir = ALGO_DIRS[args.algo]
    sys.path.insert(0, str(algo_dir))

    import torch
    import config
    import agents
    from environment import make_environment
    from rl_utils import ReplayBuffer

    apply_overrides(config, args.set)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    env, state_dim, action_dim, action_low, action_high = make_environment(
        config.ENV_NAME
    )
    # 评估用独立 env，避免 periodic eval 打断训练 episode。
    eval_env, _, _, _, _ = make_environment(config.ENV_NAME)
    agent = build_agent(
        args.algo, config, agents, state_dim, action_dim,
        action_low, action_high, device,
    )
    replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

    steps, returns, wall_clock = [], [], []
    eval_steps, eval_returns, eval_episode_returns = [], [], []
    global_step = 0
    next_eval = args.eval_every if args.eval_every > 0 else None

    print(
        f"[{args.algo} seed={args.seed}] device={device} "
        f"state_dim={state_dim} action_dim={action_dim} "
        f"episodes={args.episodes}",
        flush=True,
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(
            f"CUDA: {torch.cuda.get_device_name(device)} "
            f"({props.total_memory / 1024 ** 3:.1f} GiB, "
            f"sm_{props.major}{props.minor})",
            flush=True,
        )

    started = time.perf_counter()
    for episode in range(args.episodes):
        # DDPG 的 OU 噪声需要每个 episode 清零；TD3 的 reset_noise 是 no-op。
        if hasattr(agent, "reset_noise"):
            agent.reset_noise()

        state, _ = env.reset(seed=args.seed + episode)
        done = False
        episode_return = 0.0

        while not done:
            if args.algo == "sac":
                # SAC actor 自带随机性，不加外部噪声。
                action = agent.take_action(state)
            else:
                action = agent.take_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # truncated 只是时间上限，Bellman target 仍需 bootstrap。
            replay_buffer.add(state, action, reward, next_state, terminated)
            state = next_state
            episode_return += reward
            global_step += 1

            if replay_buffer.size() >= config.MINIMAL_SIZE:
                agent.update(replay_buffer.sample(config.BATCH_SIZE))

            # periodic 确定性评估：用于 eval 曲线和 steps-to-threshold。
            if next_eval is not None and global_step >= next_eval:
                period_returns = evaluate(
                    eval_env, agent, args.algo,
                    args.eval_episodes, args.eval_seed,
                )
                eval_steps.append(int(global_step))
                eval_returns.append(float(np.mean(period_returns)))
                eval_episode_returns.append([float(x) for x in period_returns])
                next_eval += args.eval_every

        steps.append(int(global_step))
        returns.append(float(episode_return))
        wall_clock.append(float(time.perf_counter() - started))

        if (episode + 1) % 20 == 0 or episode == 0:
            print(
                f"[{args.algo} seed={args.seed}] "
                f"ep {episode + 1}/{args.episodes} steps={global_step} "
                f"return={episode_return:.1f} "
                f"last10={np.mean(returns[-10:]):.1f} "
                f"elapsed={wall_clock[-1]:.0f}s",
                flush=True,
            )

    train_wall = time.perf_counter() - started

    # 训练结束后的最终评估：20 回合，固定 EVAL_SEED。
    final_started = time.perf_counter()
    final_returns = evaluate(
        eval_env, agent, args.algo, args.eval_episodes, args.eval_seed
    )
    final_eval_wall = time.perf_counter() - final_started

    metrics = {
        "eval_mean": float(np.mean(final_returns)),
        "eval_median": float(np.median(final_returns)),
        "eval_std": float(np.std(final_returns)),
        "eval_best": float(np.max(final_returns)),
        "eval_worst": float(np.min(final_returns)),
        "train_return_last10": float(np.mean(returns[-10:])),
        "env_steps": int(steps[-1]),
        "wall_clock_sec": float(train_wall),
        "final_eval_wall_sec": float(final_eval_wall),
    }

    payload = {
        "algo": args.algo,
        "seed": args.seed,
        "env_name": config.ENV_NAME,
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "device": str(device),
        "config": config_snapshot(config),
        "train": {
            "steps": steps,
            "returns": returns,
            "wall_clock_sec": wall_clock,
        },
        "periodic_eval": {
            "steps": eval_steps,
            "returns": eval_returns,
            "episode_returns": eval_episode_returns,
            "episodes_per_point": args.eval_episodes,
        },
        "final_eval": {
            "returns": [float(value) for value in final_returns],
            "wall_clock_sec": float(final_eval_wall),
        },
        "metrics": metrics,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.model_out:
        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algo": args.algo,
                "seed": args.seed,
                "config": payload["config"],
                "actor": agent.actor.state_dict(),
            },
            model_path,
        )

    env.close()
    eval_env.close()
    print(
        "RESULT " + json.dumps(
            {"algo": args.algo, "seed": args.seed, **metrics},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
