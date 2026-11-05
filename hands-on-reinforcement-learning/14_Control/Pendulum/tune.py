"""在调参 seed 上网格搜索"摆起 + LQR"的三个参数。

参数：能量增益 k、进入 LQR 的角度阈值 theta_enter、角速度阈值 omega_enter。
退出阈值按 1.5 倍进入阈值生成（滞回）。

搜索集：seed 20000–20019（TUNING_SEEDS）。
评估集：seed 10000–10019 只在 evaluate.py 里用一次，不参与选参。

打分按字典序：严格成功率 -> 回报成功率 -> 平均回报。成功率并列时
平均回报高的优先，避免选到"能稳住但摆起很慢、代价很大"的参数。

跑法：
    python -I 14_Control/Pendulum/tune.py
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from model import continuous_linearization, lqr_discrete, zoh_discretize
from sim import evaluate_controller, initial_states
from swingup import TUNING_SEEDS, build_switched

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

K_GRID = (0.1, 0.2, 0.3, 0.5, 0.8)
THETA_ENTER_GRID = (0.3, 0.5, 0.7, 0.9)
OMEGA_ENTER_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)


def nominal_gain(q_theta=1.0, q_theta_dot=0.1, r=0.001):
    A, B = continuous_linearization()
    Ad, Bd = zoh_discretize(A, B)
    K, _ = lqr_discrete(Ad, Bd, np.diag([q_theta, q_theta_dot]), np.array([[r]]))
    return K


def main():
    parser = argparse.ArgumentParser(description="网格搜索摆起 + LQR 的切换参数")
    parser.add_argument("--seeds", type=int, default=len(TUNING_SEEDS))
    parser.add_argument("--out", default=str(RESULTS / "control_tuning.json"))
    args = parser.parse_args()

    seeds = TUNING_SEEDS[: args.seeds]
    states = initial_states(seeds)
    K = nominal_gain()

    print(f"调参起点：seed {seeds[0]}–{seeds[-1]}（{len(seeds)} 个），不包含评估集")
    print(f"K = {np.round(K.ravel(), 4)}")
    print(
        f"网格：k {K_GRID} × theta_enter {THETA_ENTER_GRID} "
        f"× omega_enter {OMEGA_ENTER_GRID} = "
        f"{len(K_GRID) * len(THETA_ENTER_GRID) * len(OMEGA_ENTER_GRID)} 组"
    )

    started = time.perf_counter()
    records = []
    for k, theta_enter, omega_enter in product(
        K_GRID, THETA_ENTER_GRID, OMEGA_ENTER_GRID
    ):
        params = {
            "k": k,
            "theta_enter": theta_enter,
            "omega_enter": omega_enter,
            "theta_exit": theta_enter * 1.5,
            "omega_exit": omega_enter * 1.5,
        }
        controller = build_switched(K, params)
        _, summary = evaluate_controller(controller, seeds, states=states)
        records.append({"params": params, "summary": summary})

    records.sort(
        key=lambda r: (
            r["summary"]["success_strict_rate"],
            r["summary"]["success_return_rate"],
            r["summary"]["mean_return"],
        ),
        reverse=True,
    )

    elapsed = time.perf_counter() - started
    print(f"完成，用时 {elapsed:.1f} s\n")
    print(
        f"{'k':>5} {'theta_in':>9} {'omega_in':>9} "
        f"{'严格成功率':>10} {'回报成功率':>10} {'平均回报':>10}"
    )
    for record in records[:10]:
        p = record["params"]
        s = record["summary"]
        print(
            f"{p['k']:>5.2f} {p['theta_enter']:>9.2f} {p['omega_enter']:>9.2f} "
            f"{s['success_strict_rate']:>10.2f} {s['success_return_rate']:>10.2f} "
            f"{s['mean_return']:>10.2f}"
        )

    best = records[0]
    output = {
        "tuning_seeds": seeds,
        "grid": {
            "k": list(K_GRID),
            "theta_enter": list(THETA_ENTER_GRID),
            "omega_enter": list(OMEGA_ENTER_GRID),
        },
        "best_params": best["params"],
        "best_summary": best["summary"],
        "all": records,
        "wall_clock_sec": elapsed,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最佳参数已写入 {out_path}")


if __name__ == "__main__":
    main()
