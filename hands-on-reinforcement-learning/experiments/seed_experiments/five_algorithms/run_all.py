"""五算法统一实验编排：DDPG / TD3 / SAC（off-policy）+ PPO / TRPO（on-policy）。

固定 seed 协议（所有算法共用）：
- seeds 0..N-1：seed 决定网络初始化、训练时每个回合的 env.reset(seed + episode)；
- 评估初始状态固定：EVAL_SEED..EVAL_SEED+19，所有算法、所有 seed、所有评估点完全相同，
  因此同一 seed、同一评估点上算法之间的差异不含初始状态抽样噪声；
- 预算：off-policy 200 回合 = 40k env steps；on-policy 2000 回合 = 400k env steps；
- 评估只用确定性策略；训练回报含探索噪声，不用于跨算法比较。

输出（experiments/seed_experiments/five_algorithms/results/）：
    raw/{algo}_seed{seed}.json   单次运行的完整记录
    models/{algo}_seed{seed}.pt  actor 权重
    protocol.json                本次实验的参数快照

用法：
    python experiments/seed_experiments/five_algorithms/run_all.py                 # seeds 0..9，已有结果会跳过
    python experiments/seed_experiments/five_algorithms/run_all.py --seeds 0 1 2 --algos ppo trpo --force
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]

OFF_POLICY = ["ddpg", "td3", "sac"]
ON_POLICY = ["ppo", "trpo"]
ALGOS = OFF_POLICY + ON_POLICY


def parse_args():
    parser = argparse.ArgumentParser(description="Run all five algorithms.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    parser.add_argument("--off-episodes", type=int, default=200,
                        help="off-policy 回合数；200 回合 = 40k 步")
    parser.add_argument("--on-episodes", type=int, default=2000,
                        help="on-policy 回合数；2000 回合 = 400k 步")
    parser.add_argument("--off-eval-every", type=int, default=2000)
    parser.add_argument("--on-eval-every", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--outdir", default=str(EXP_DIR / "results"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_command(args, algo, seed, raw_dir, model_dir):
    on_policy = algo in ON_POLICY
    script = EXP_DIR / ("run_one_onpolicy.py" if on_policy else "run_one.py")
    return [
        sys.executable, "-I", str(script),
        "--algo", algo, "--seed", str(seed),
        "--episodes", str(args.on_episodes if on_policy else args.off_episodes),
        "--eval-every", str(args.on_eval_every if on_policy else args.off_eval_every),
        "--eval-episodes", str(args.eval_episodes),
        "--eval-seed", str(args.eval_seed),
        "--device", args.device,
        "--out", str(raw_dir / f"{algo}_seed{seed}.json"),
        "--model-out", str(model_dir / f"{algo}_seed{seed}.pt"),
    ]


def run_job(job):
    algo, seed, cmd, log_path = job
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        code = subprocess.run(cmd, cwd=str(REPO), env=env,
                              stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.perf_counter() - started
    print(f"[{'ok' if code == 0 else 'FAIL'}] {algo} seed={seed} "
          f"({elapsed:.0f}s)", flush=True)
    return code


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    raw_dir, model_dir, log_dir = out_dir / "raw", out_dir / "models", out_dir / "logs"
    for path in (raw_dir, model_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    jobs = []
    for algo in args.algos:
        for seed in args.seeds:
            if (raw_dir / f"{algo}_seed{seed}.json").exists() and not args.force:
                print(f"[skip] {algo} seed={seed}", flush=True)
                continue
            jobs.append((algo, seed, build_command(args, algo, seed, raw_dir, model_dir),
                         log_dir / f"{algo}_seed{seed}.log"))

    # 慢任务（SAC）排在前面，避免最后只剩一个长尾。
    jobs.sort(key=lambda job: ALGOS.index(job[0]) != 2)
    print(f"{len(jobs)} runs, {args.jobs} parallel", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(run_job, jobs))
    if any(codes):
        raise SystemExit("有运行失败，见 results/logs/")

    protocol = {
        "seeds": args.seeds,
        "algos": args.algos,
        "off_policy_steps": args.off_episodes * 200,
        "on_policy_steps": args.on_episodes * 200,
        "off_eval_every": args.off_eval_every,
        "on_eval_every": args.on_eval_every,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "device": args.device,
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
