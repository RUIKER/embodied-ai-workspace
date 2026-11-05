"""扰动实验：模型误差与执行延迟下，经典控制器退化多快。

两组扰动：

1. 参数扰动：被控对象的质量 m、长度 l 分别乘 0.7 / 0.85 / 1.0 / 1.15 / 1.3，
   控制器**仍按名义参数设计，不重调**。这是"模型不准"的简化代理，
   不是 sim2real。
2. 执行延迟：动作晚 0 / 1 / 2 / 3 步生效（每步 0.05 s）。
   延迟在频域里是纯相位滞后，吃掉相位裕度；这个系统本身有不稳定极点，
   裕度本来就小。规划期试算只测了 LQR 单独，这里按计划改成
   "摆起 + LQR" 也一起测。

起点仍是评估集 seed 10000–10019，每个条件跑同样的 20 个起点。

跑法：
    python -I 14_Control/Pendulum/robustness.py
    python -I 14_Control/Pendulum/robustness.py --plot
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from model import continuous_linearization, lqr_discrete, zoh_discretize
from sim import evaluate_controller, initial_states
from swingup import DEFAULT_PARAMS, EVAL_SEEDS, LQRController, build_switched

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

PARAM_RATIOS = (0.7, 0.85, 1.0, 1.15, 1.3)
DELAYS = (0, 1, 2, 3)


def nominal_gain(q_theta=1.0, q_theta_dot=0.1, r=0.001):
    A, B = continuous_linearization()
    Ad, Bd = zoh_discretize(A, B)
    K, _ = lqr_discrete(Ad, Bd, np.diag([q_theta, q_theta_dot]), np.array([[r]]))
    return K


def load_tuned_params(path=None):
    path = Path(path) if path else RESULTS / "control_tuning.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["best_params"]
    return dict(DEFAULT_PARAMS)


def make_controllers(K, tuned):
    return {
        "lqr": lambda: LQRController(K),
        "lqr_swingup": lambda: build_switched(K, tuned),
    }


def run_all(controllers, seeds, states):
    records = []
    for name, factory in controllers.items():
        for ratio in PARAM_RATIOS:
            for param in ("mass", "length"):
                plant = {"m": 1.0 * ratio} if param == "mass" else {"l": 1.0 * ratio}
                _, summary = evaluate_controller(
                    factory(), seeds, states=states, **plant
                )
                records.append(
                    {
                        "controller": name,
                        "condition": param,
                        "value": ratio,
                        **summary,
                    }
                )
        for delay in DELAYS:
            _, summary = evaluate_controller(
                factory(), seeds, states=states, delay_steps=delay
            )
            records.append(
                {
                    "controller": name,
                    "condition": "delay_steps",
                    "value": delay,
                    **summary,
                }
            )
    return records


def write_csv(records, path):
    fields = [
        "controller",
        "condition",
        "value",
        "n",
        "success_strict_rate",
        "success_return_rate",
        "mean_return",
        "median_return",
        "mean_time_to_upright",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})


def plot(records, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    colors = {"lqr": "#B45309", "lqr_swingup": "#1D4ED8"}
    markers = {"lqr": "o", "lqr_swingup": "s"}
    labels = {"lqr": "LQR only", "lqr_swingup": "swing-up + LQR"}
    linestyles = {"mass": "-", "length": ":"}

    for name in ("lqr", "lqr_swingup"):
        for param in ("mass", "length"):
            points = [
                (record["value"], record["success_strict_rate"])
                for record in records
                if record["controller"] == name and record["condition"] == param
            ]
            xs, ys = zip(*points)
            axes[0].plot(
                xs,
                ys,
                color=colors[name],
                marker=markers[name],
                linestyle=linestyles[param],
                label=f"{labels[name]} · {param}",
            )

        points = [
            (record["value"], record["success_strict_rate"])
            for record in records
            if record["controller"] == name and record["condition"] == "delay_steps"
        ]
        xs, ys = zip(*points)
        axes[1].plot(xs, ys, color=colors[name], marker=markers[name], label=labels[name])

    axes[0].set_xlabel("mass / length scale (controller not retuned)")
    axes[0].set_ylabel("strict success rate")
    axes[1].set_xlabel("actuation delay (steps of 0.05 s)")
    axes[1].set_ylabel("strict success rate")
    for ax in axes:
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    print(f"图已保存到 {path}")


def main():
    parser = argparse.ArgumentParser(description="参数扰动与执行延迟")
    parser.add_argument("--tuning", default=None)
    parser.add_argument("--out", default=str(RESULTS / "control_robustness.csv"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    seeds = EVAL_SEEDS
    states = initial_states(seeds)
    K = nominal_gain()
    tuned = load_tuned_params(args.tuning)
    controllers = make_controllers(K, tuned)

    print(f"评估起点 seed {seeds[0]}–{seeds[-1]}；控制器不随扰动重调")
    records = run_all(controllers, seeds, states)

    print(
        f"{'controller':>12} {'condition':>12} {'value':>6} "
        f"{'严格成功率':>10} {'回报成功率':>10} {'平均回报':>10}"
    )
    for record in records:
        print(
            f"{record['controller']:>12} {record['condition']:>12} "
            f"{record['value']:>6.2f} {record['success_strict_rate']:>10.2f} "
            f"{record['success_return_rate']:>10.2f} {record['mean_return']:>10.2f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(records, out_path)
    print(f"\n结果已写入 {out_path}")

    if args.plot:
        plot(records, out_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
