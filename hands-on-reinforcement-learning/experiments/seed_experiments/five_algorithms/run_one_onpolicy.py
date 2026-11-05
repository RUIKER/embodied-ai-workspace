"""单次 on-policy 训练驱动：PPO / TRPO on Pendulum-v1。

与 run_one.py 的 off-policy 驱动保持同一套协议：
- seed 同时用于 PyTorch 初始化、每个 episode 的 env.reset(seed+episode)；
- 每 eval_every 个 env steps 做一次确定性评估：每个评估点都用同一组 20 个
  初始状态（EVAL_SEED..EVAL_SEED+19），所有算法、所有 seed、所有评估点共用；
- 训练结束后再用同一组初始状态做最终评估。

采集方式与 config.EPISODES_PER_UPDATE 一致：攒够若干完整回合再更新一次。
`dones` 只记 terminated（用于 bootstrap），`ends` 记回合边界（用于切断 GAE）。

用法：
    python experiments/seed_experiments/five_algorithms/run_one_onpolicy.py --algo ppo --seed 0 --episodes 2000 \
        --gamma 0.99 --eval-every 10000 --out experiments/seed_experiments/five_algorithms/results_onpolicy/raw/ppo_seed0.json
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
    "ppo": REPO / "08_PPO" / "Pendulum",
    "trpo": REPO / "07_TRPO" / "Pendulum",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one fixed-seed on-policy RL experiment.")
    parser.add_argument("--algo", required=True, choices=sorted(ALGO_DIRS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=10000,
                        help="每多少个 env steps 做一次 periodic eval；0 表示关闭")
    parser.add_argument("--device", default="cpu",
                        help="网络很小，CPU 比 GPU 快一个数量级（400k 步约 30 s）")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="覆盖 config 里的超参，例如 --set KL_CONSTRAINT=0.02")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-out", default=None)
    return parser.parse_args()


def set_seed(seed):
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
    if algo == "ppo":
        return agents.PPOContinuous(
            state_dim, config.HIDDEN_DIM, action_dim, action_low, action_high,
            config.ACTOR_LR, config.CRITIC_LR, config.LMBDA, config.EPOCHS,
            config.EPS, config.GAMMA, config.REWARD_SHIFT, config.REWARD_SCALE,
            device, config.MINIBATCH_SIZE, config.MAX_GRAD_NORM,
        )
    return agents.TRPOContinuous(
        state_dim, action_dim, config.HIDDEN_DIM, action_low, action_high,
        config.LMBDA, config.KL_CONSTRAINT, config.ALPHA, config.CRITIC_LR,
        config.GAMMA, config.REWARD_SHIFT, config.REWARD_SCALE, device,
        config.CRITIC_EPOCHS, config.MINIBATCH_SIZE,
    )


def deterministic_action(agent, state):
    """PPO/TRPO 的确定性动作：高斯策略的均值 mu，不加采样噪声。"""
    import torch

    state_t = torch.as_tensor(
        np.asarray(state), dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    with torch.no_grad():
        mean, _ = agent.actor(state_t)
    return mean.squeeze(0).cpu().numpy()


def evaluate(env, agent, num_episodes, seed):
    returns = []
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            action = deterministic_action(agent, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(float(episode_return))
    return np.asarray(returns, dtype=np.float32)


def config_snapshot(config):
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

    apply_overrides(config, args.set)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    env, state_dim, action_dim, action_low, action_high = make_environment(
        config.ENV_NAME
    )
    eval_env, _, _, _, _ = make_environment(config.ENV_NAME)
    agent = build_agent(
        args.algo, config, agents, state_dim, action_dim,
        action_low, action_high, device,
    )

    steps, returns, wall_clock = [], [], []
    eval_steps, eval_returns, eval_episode_returns = [], [], []
    global_step = 0
    next_eval = args.eval_every if args.eval_every > 0 else None
    per_update = config.EPISODES_PER_UPDATE

    print(
        f"[{args.algo} seed={args.seed}] device={device} gamma={config.GAMMA} "
        f"episodes={args.episodes} per_update={per_update}",
        flush=True,
    )

    def periodic_eval():
        returns_ = evaluate(eval_env, agent, args.eval_episodes, args.eval_seed)
        eval_steps.append(int(global_step))
        eval_returns.append(float(np.mean(returns_)))
        eval_episode_returns.append([float(x) for x in returns_])

    started = time.perf_counter()
    episode = 0
    while episode < args.episodes:
        transition_dict = {
            "states": [], "actions": [], "next_states": [],
            "rewards": [], "dones": [], "ends": [],
        }
        for _ in range(min(per_update, args.episodes - episode)):
            state, _ = env.reset(seed=args.seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                transition_dict["states"].append(state)
                transition_dict["actions"].append(action)
                transition_dict["next_states"].append(next_state)
                transition_dict["rewards"].append(reward)
                transition_dict["dones"].append(terminated)
                transition_dict["ends"].append(done)

                state = next_state
                episode_return += reward
                global_step += 1

                if next_eval is not None and global_step >= next_eval:
                    periodic_eval()
                    next_eval += args.eval_every

            steps.append(int(global_step))
            returns.append(float(episode_return))
            wall_clock.append(float(time.perf_counter() - started))
            episode += 1

            if episode % 100 == 0 or episode == 1:
                print(
                    f"[{args.algo} seed={args.seed}] "
                    f"ep {episode}/{args.episodes} steps={global_step} "
                    f"last10={np.mean(returns[-10:]):.1f} "
                    f"elapsed={wall_clock[-1]:.0f}s",
                    flush=True,
                )

        agent.update(transition_dict)

    train_wall = time.perf_counter() - started

    final_started = time.perf_counter()
    final_returns = evaluate(
        eval_env, agent, args.eval_episodes, args.eval_seed
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
