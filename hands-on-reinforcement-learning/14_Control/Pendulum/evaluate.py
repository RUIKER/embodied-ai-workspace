"""按 experiments 的固定起点协议评估经典控制器。

评估集：seed 10000–10019，与 experiments/seed_experiments/five_algorithms
完全相同。这一批起点只用来出最终数字，不参与选参（选参见 tune.py）。

跑法：
    python -I 14_Control/Pendulum/evaluate.py --controller lqr
    python -I 14_Control/Pendulum/evaluate.py --controller lqr_swingup
    python -I 14_Control/Pendulum/evaluate.py --controller lqr_swingup --compare-sac

--compare-sac 读取 experiments 里 10 个 seed 的 SAC 最终评估回报，只做
名义条件下的并排展示；SAC 的"严格成功率"需要逐回合状态，现有结果里没有，
所以只报回报口径。扰动和延迟下的 SAC 需要重新训练并存权重（W4 可选事项）。
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from model import continuous_linearization, lqr_discrete, zoh_discretize
from sim import RETURN_THRESHOLD, evaluate_controller, initial_states
from swingup import DEFAULT_PARAMS, EVAL_SEEDS, LQRController, build_switched

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SAC_RAW = REPO / "experiments" / "seed_experiments" / "five_algorithms" / "results" / "raw"


def nominal_gain(q_theta=1.0, q_theta_dot=0.1, r=0.001):
    A, B = continuous_linearization()
    Ad, Bd = zoh_discretize(A, B)
    K, _ = lqr_discrete(Ad, Bd, np.diag([q_theta, q_theta_dot]), np.array([[r]]))
    return K


def load_tuned_params(path=None):
    """优先用 tune.py 的结果；没有就用 swingup.DEFAULT_PARAMS。"""
    path = Path(path) if path else RESULTS / "control_tuning.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["best_params"], str(path)
    return dict(DEFAULT_PARAMS), None


def print_rows(rows):
    print(
        f"{'seed':>6} {'回报':>10} {'严格成功':>8} {'回报成功':>8} "
        f"{'摆起时间/s':>10} {'末角/rad':>10}"
    )
    for row in rows:
        t = row["time_to_upright"]
        t_str = f"{t:.2f}" if t is not None else "—"
        print(
            f"{row['seed']:>6} {row['return']:>10.2f} "
            f"{'是' if row['success_strict'] else '否':>8} "
            f"{'是' if row['success_return'] else '否':>8} "
            f"{t_str:>10} {row['final_angle']:>10.3f}"
        )


def print_summary(name, summary):
    time_to_upright = summary["mean_time_to_upright"]
    time_str = "—" if time_to_upright is None else f"{time_to_upright:.2f} s"
    print(
        f"[{name}] 严格成功率 {summary['success_strict_rate']:.2f}  "
        f"回报成功率 {summary['success_return_rate']:.2f}  "
        f"平均回报 {summary['mean_return']:.2f}  "
        f"中位数回报 {summary['median_return']:.2f}  "
        f"平均摆起时间 {time_str}"
    )


def load_sac_final_returns():
    """读取 SAC 10 个 seed 的最终评估（20 个固定起点上的逐回合回报）。"""
    paths = sorted(glob.glob(str(SAC_RAW / "sac_seed*.json")))
    if not paths:
        return None
    per_seed = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        per_seed.append(np.asarray(data["final_eval"]["returns"], dtype=float))
    return per_seed


def print_sac_comparison():
    per_seed = load_sac_final_returns()
    if per_seed is None:
        print("\n未找到 SAC 结果，跳过对比。")
        return None
    all_returns = np.concatenate(per_seed)
    per_seed_means = np.array([arr.mean() for arr in per_seed])
    out = {
        "n_seeds": len(per_seed),
        "n_episodes": int(all_returns.size),
        "mean_return": float(all_returns.mean()),
        "median_per_seed_return": float(np.median(per_seed_means)),
        "success_return_rate": float(np.mean(all_returns >= RETURN_THRESHOLD)),
    }
    print("\nSAC（experiments 的 10 个 seed，名义参数，同一批 20 个起点）：")
    print(
        f"  逐 seed 平均回报中位数 {out['median_per_seed_return']:.2f}  "
        f"全部 {len(per_seed)}×20 回合平均回报 {out['mean_return']:.2f}  "
        f"回报成功率 {out['success_return_rate']:.2f}"
    )
    print("  注意：SAC 的严格成功率（末 2.5 s |theta| < 0.15）需要逐回合状态，")
    print("       现有 JSON 只存了回报，所以这一列不给 SAC 数字。")
    return out


def main():
    parser = argparse.ArgumentParser(description="评估 LQR / 摆起 + LQR")
    parser.add_argument(
        "--controller", choices=("lqr", "lqr_swingup", "both"), default="both"
    )
    parser.add_argument("--compare-sac", action="store_true", help="并行展示 SAC 的回报口径")
    parser.add_argument("--tuning", default=None, help="tune.py 输出的 JSON 路径")
    parser.add_argument("--out", default=None, help="结果 JSON 路径（默认写入 results/）")
    args = parser.parse_args()

    seeds = EVAL_SEEDS
    states = initial_states(seeds)
    K = nominal_gain()
    tuned, tuned_from = load_tuned_params(args.tuning)

    print(f"评估起点：seed {seeds[0]}–{seeds[-1]}（{len(seeds)} 个，与 experiments 相同）")
    print(f"K = {np.round(K.ravel(), 4)}")
    if tuned_from:
        print(f"切换参数来自 {tuned_from}")
    else:
        print(f"切换参数使用默认值 {DEFAULT_PARAMS}（尚未运行 tune.py）")
    print(f"回报阈值：{RETURN_THRESHOLD:.0f}；严格成功：末 2.5 s |theta| < 0.15 rad\n")

    payload = {"eval_seeds": seeds, "K": K.ravel().tolist(), "controllers": {}}

    if args.controller in ("lqr", "both"):
        print("=" * 64)
        print("LQR 单独（无摆起）")
        print("=" * 64)
        rows, summary = evaluate_controller(LQRController(K), seeds, states=states)
        print_rows(rows)
        print_summary("LQR", summary)
        payload["controllers"]["lqr"] = {"rows": rows, "summary": summary}
        print()

    if args.controller in ("lqr_swingup", "both"):
        print("=" * 64)
        print("能量摆起 + LQR 切换")
        print("=" * 64)
        controller = build_switched(K, tuned)
        rows, summary = evaluate_controller(controller, seeds, states=states)
        print_rows(rows)
        print_summary("摆起+LQR", summary)
        payload["controllers"]["lqr_swingup"] = {
            "params": tuned,
            "rows": rows,
            "summary": summary,
        }
        print()

    if args.compare_sac:
        sac = print_sac_comparison()
        if sac:
            payload["sac_reference"] = sac

    out_path = Path(args.out) if args.out else RESULTS / "control_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
