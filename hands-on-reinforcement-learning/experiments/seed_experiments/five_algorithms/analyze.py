"""读取 results/raw 下的运行记录，生成表格和三张图。

固定 seed 实验的展示原则：
- seed 是实验单位，不是噪声：每条细线 / 每个点 / 每个格子都是一个具体的 seed，
  聚合统计（中位数、IQR）只是辅助；
- 所有评估点都用同一组 20 个初始状态，所以同一 seed 下算法之间的差异不含初始状态抽样噪声；
- 前 3 个 seed（0–2）在调 PPO / TRPO 配置时看过，其余 seed 是留出集，图里用虚线分开。

输出（results/）：
    fig1_learning_curves.png/.pdf   学习曲线：五算法中位数 + 每个算法的逐 seed 曲线
    fig2_final_by_seed.png/.pdf     最终成绩的 seed × 算法网格 + 达到阈值所需步数
    fig3_eval_states.png/.pdf       20 个固定评估初始状态上的逐状态回报（诊断图）
    per_seed.csv / summary.csv / tables.md

用法：
    python analyze.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

EXP_DIR = Path(__file__).resolve().parent

ALGOS = ["ddpg", "td3", "sac", "ppo", "trpo"]
OFF_POLICY = {"ddpg", "td3", "sac"}
LABELS = {"ddpg": "DDPG", "td3": "TD3", "sac": "SAC", "ppo": "PPO", "trpo": "TRPO"}
# 参考色板里经 validate_palette.js 做过 5 色全对校验（无 FAIL）的一组；
# 色盲区分度只到 floor 档，所以同时用线型 / marker / 直接标注做第二重编码。
COLORS = {"ddpg": "#2a78d6", "td3": "#eda100", "sac": "#1baf7a",
          "ppo": "#e34948", "trpo": "#008300"}
MARKERS = {"ddpg": "o", "td3": "s", "sac": "D", "ppo": "^", "trpo": "v"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

RANDOM_RETURN = -1200.0   # 随机策略大致水平，仅作参照线
THRESHOLD = -200.0        # 样本效率阈值：20 个固定初始状态的平均回报
DEV_SEEDS = {0, 1, 2}     # 调 PPO / TRPO 配置时看过的 seed
FAIL_LEVEL = -400.0       # 低于该值的 seed 在图里标出编号


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze five-algorithm runs.")
    parser.add_argument("--results", default=str(EXP_DIR / "results"))
    return parser.parse_args()


def load_runs(raw_dir):
    runs = {}
    for path in sorted(Path(raw_dir).glob("*_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        runs[(data["algo"], data["seed"])] = data
    return runs


def first_hit(run, threshold=THRESHOLD):
    """第一次评估均值 >= threshold 的 env step；从未达到返回 None。"""
    for step, value in zip(run["periodic_eval"]["steps"], run["periodic_eval"]["returns"]):
        if value >= threshold:
            return step
    return None


def final_mean(run):
    return float(np.mean(run["final_eval"]["returns"]))


def quartiles(values):
    values = np.asarray(values, dtype=float)
    return np.percentile(values, [25, 50, 75])


def style_axis(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def step_ticks(ax, upper):
    ticks = [t for t in (2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 4e5) if t <= upper]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t / 1000)}k" for t in ticks])


def eval_matrix(runs, algo, seeds):
    steps = np.asarray(runs[(algo, seeds[0])]["periodic_eval"]["steps"])
    values = np.asarray([runs[(algo, s)]["periodic_eval"]["returns"][:len(steps)]
                         for s in seeds])
    return steps, values


# --------------------------------------------------------------------------- 表格
def write_tables(runs, seeds, out_dir):
    per_seed = []
    for algo in ALGOS:
        for seed in seeds:
            run = runs[(algo, seed)]
            per_seed.append({
                "algo": algo, "seed": seed, "final_eval": final_mean(run),
                "steps_to_thr": first_hit(run),
                "train_last10": run["metrics"]["train_return_last10"],
                "wall_clock_sec": run["metrics"]["wall_clock_sec"],
            })
    with open(out_dir / "per_seed.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)

    summary = []
    for algo in ALGOS:
        rows = [r for r in per_seed if r["algo"] == algo]
        finals = np.array([r["final_eval"] for r in rows])
        q1, med, q3 = quartiles(finals)
        hits = [r["steps_to_thr"] for r in rows if r["steps_to_thr"] is not None]
        held = np.array([r["final_eval"] for r in rows if r["seed"] not in DEV_SEEDS])
        summary.append({
            "algo": algo, "n": len(rows), "median": med, "q1": q1, "q3": q3,
            "min": finals.min(), "max": finals.max(),
            "held_out_median": float(np.median(held)) if len(held) else float("nan"),
            "n_success": int((finals >= THRESHOLD).sum()),
            "n_reached": len(hits),
            "median_steps_to_thr": float(np.median(hits)) if len(hits) > len(rows) / 2 else None,
            "wall_median": float(np.median([r["wall_clock_sec"] for r in rows])),
            "budget": int(runs[(algo, seeds[0])]["metrics"]["env_steps"]),
        })
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        f"seeds = {seeds}（n={len(seeds)}）；阈值 = 20 个固定初始状态平均回报 ≥ {THRESHOLD:.0f}", "",
        "| 算法 | 预算 | 最终评估中位数 [Q1, Q3] | 最差 / 最好 seed | 留出 seed 中位数 | "
        f"最终 ≥ {THRESHOLD:.0f} 的 seed 数 | 首次达到阈值的步数（中位数） | 耗时中位数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        steps = (f"{row['median_steps_to_thr'] / 1000:.0f}k" if row["median_steps_to_thr"]
                 is not None else "未达到")
        lines.append(
            f"| {LABELS[row['algo']]} | {row['budget'] // 1000}k | "
            f"{row['median']:.1f} [{row['q1']:.1f}, {row['q3']:.1f}] | "
            f"{row['min']:.0f} / {row['max']:.0f} | {row['held_out_median']:.1f} | "
            f"{row['n_success']}/{row['n']} | {steps} ({row['n_reached']}/{row['n']} 个 seed 达到) | "
            f"{row['wall_median']:.0f} s |")
    lines += ["", "逐 seed 最终评估：", "",
              "| 算法 | " + " | ".join(f"s{s}" for s in seeds) + " |",
              "| --- |" + " ---: |" * len(seeds)]
    for algo in ALGOS:
        values = [final_mean(runs[(algo, s)]) for s in seeds]
        lines.append(f"| {LABELS[algo]} | " + " | ".join(f"{v:.0f}" for v in values) + " |")
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------------- 图 1：曲线
def plot_learning_curves(runs, seeds, out_path):
    fig = plt.figure(figsize=(14, 9), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 5, height_ratios=[1.55, 1], hspace=0.5, wspace=0.12)

    ax = fig.add_subplot(grid[0, :])
    style_axis(ax)
    ends = []
    for algo in ALGOS:
        steps, values = eval_matrix(runs, algo, seeds)
        q1, med, q3 = np.percentile(values, [25, 50, 75], axis=0)
        ax.fill_between(steps, q1, q3, color=COLORS[algo], alpha=0.13, linewidth=0)
        ax.plot(steps, med, color=COLORS[algo], linewidth=2.2,
                linestyle="-" if algo in OFF_POLICY else (0, (5, 2)),
                marker=MARKERS[algo], markersize=5, markevery=1 if algo in OFF_POLICY else 3,
                markeredgecolor=SURFACE, markeredgewidth=1, label=LABELS[algo])
        ends.append([steps[-1], med[-1], algo])
    # 线尾标注：同一横坐标上的标签按 y 排开，避免 TD3 / SAC / DDPG 相撞。
    for x_end in sorted({e[0] for e in ends}):
        group = sorted((e for e in ends if e[0] == x_end), key=lambda e: e[1])
        for low, high in zip(group, group[1:]):
            high[1] = max(high[1], low[1] + 75)
    for x_end, y_label, algo in ends:
        ax.annotate(LABELS[algo], (x_end, y_label), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=10,
                    fontweight="bold", color=INK)
    ax.axhline(RANDOM_RETURN, color=INK_2, linestyle=":", linewidth=1)
    ax.text(2.1e3, RANDOM_RETURN + 25, "random policy ≈ -1200", fontsize=8, color=INK_2)
    ax.axhline(THRESHOLD, color=INK_2, linestyle=(0, (1, 3)), linewidth=1)
    ax.text(2.1e3, THRESHOLD + 25, f"threshold {THRESHOLD:.0f}", fontsize=8, color=INK_2)
    ax.axvline(40000, color=GRID, linewidth=1.6, zorder=0)
    ax.text(41500, -1620, "off-policy budget (40k)", fontsize=8, color=INK_2, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(2e3, 8e5)
    ax.set_ylim(-1650, 60)
    step_ticks(ax, 4e5)
    ax.set_xlabel("Environment steps (log scale)", color=INK_2, fontsize=10)
    ax.set_ylabel("Eval return, mean of 20 fixed start states", color=INK_2, fontsize=10)
    ax.set_title(f"Median across {len(seeds)} seeds, band = interquartile range"
                 "   (solid: off-policy, dashed: on-policy)",
                 loc="left", color=INK, fontsize=11.5)

    shared = None
    for column, algo in enumerate(ALGOS):
        small = fig.add_subplot(grid[1, column], sharey=shared)
        shared = shared or small
        style_axis(small)
        steps, values = eval_matrix(runs, algo, seeds)
        for seed, curve in zip(seeds, values):
            small.plot(steps, curve, color=COLORS[algo], alpha=0.35, linewidth=1)
            if curve[-1] < FAIL_LEVEL:
                small.annotate(f"s{seed}", (steps[-1], curve[-1]), xytext=(3, 0),
                               textcoords="offset points", fontsize=7.5,
                               color=INK, va="center")
        small.plot(steps, np.median(values, axis=0), color=COLORS[algo], linewidth=2.2)
        small.axhline(THRESHOLD, color=INK_2, linestyle=(0, (1, 3)), linewidth=0.9)
        small.set_xlim(0, steps[-1] * 1.12)
        small.set_ylim(-1650, 60)
        small.set_title(f"{LABELS[algo]}  ({steps[-1] // 1000}k steps)", loc="left",
                        color=INK, fontsize=10.5)
        small.set_xticks([0, steps[-1] // 2, steps[-1]])
        small.set_xticklabels(["0", f"{steps[-1] // 2000}k", f"{steps[-1] // 1000}k"])
        if column:
            plt.setp(small.get_yticklabels(), visible=False)
        else:
            small.set_ylabel("Eval return", color=INK_2, fontsize=9.5)
    fig.text(0.01, 0.43, f"Bottom row: one thin line = one seed (index shown if final < {FAIL_LEVEL:.0f}), "
             "bold = median.  Every evaluation point uses the same 20 start states.",
             ha="left", fontsize=9.5, color=INK_2)
    fig.suptitle(f"Pendulum-v1 · five algorithms · seeds {seeds[0]}–{seeds[-1]} fixed",
                 x=0.01, ha="left", fontsize=13.5, color=INK, y=0.985)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- 图 2：seed × 算法网格
def text_color(rgba):
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return INK if luminance > 0.55 else "#ffffff"


def plot_final_by_seed(runs, seeds, out_path):
    fig, (ax_grid, ax_steps) = plt.subplots(
        1, 2, figsize=(14, 4.6), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.28})

    cmap = LinearSegmentedColormap.from_list("ret", ["#f2f6fb", "#9ec3ec", "#2a78d6", "#0e3f7a"])
    low, high = -300.0, -120.0   # 窄色阶：突出偏离 ≈ -125 的 seed，< -300 统一最深
    matrix = np.array([[final_mean(runs[(a, s)]) for s in seeds] for a in ALGOS])
    ax_grid.set_facecolor(SURFACE)
    ax_grid.imshow(np.clip(matrix, low, high), cmap=cmap.reversed(),
                   vmin=low, vmax=high, aspect="auto")
    for row, algo in enumerate(ALGOS):
        for col, seed in enumerate(seeds):
            value = matrix[row, col]
            rgba = cmap.reversed()((np.clip(value, low, high) - low) / (high - low))
            ax_grid.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=9,
                         color=text_color(rgba))
    ax_grid.set_xticks(range(len(seeds)))
    ax_grid.set_xticklabels([f"s{s}" for s in seeds], fontsize=9.5, color=INK_2)
    ax_grid.set_yticks(range(len(ALGOS)))
    ax_grid.set_yticklabels([LABELS[a] for a in ALGOS], fontsize=10.5, color=INK)
    ax_grid.tick_params(length=0)
    for side in ax_grid.spines.values():
        side.set_visible(False)
    ax_grid.axhline(2.5, color=SURFACE, linewidth=5)
    dev_edge = max((i for i, s in enumerate(seeds) if s in DEV_SEEDS), default=None)
    if dev_edge is not None and dev_edge < len(seeds) - 1:
        ax_grid.axvline(dev_edge + 0.5, color=INK, linewidth=1.2, linestyle=(0, (3, 2)))
        ax_grid.text(dev_edge / 2, -0.62, "seeds seen while\ntuning PPO/TRPO", ha="center",
                     fontsize=8, color=INK_2, va="bottom")
        ax_grid.text((dev_edge + len(seeds)) / 2, -0.62, "held-out seeds", ha="center",
                     fontsize=8, color=INK_2, va="bottom")
    ax_grid.set_title("Final eval return per seed (20 fixed start states)", loc="left",
                      color=INK, fontsize=11.5, pad=44)

    style_axis(ax_steps, grid_axis="x")
    ax_steps.grid(axis="y", visible=False)
    not_reached_x = 9e5
    for row, algo in enumerate(ALGOS):
        hits = [first_hit(runs[(algo, s)]) for s in seeds]
        reached = [h for h in hits if h is not None]
        jitter = np.linspace(-0.16, 0.16, len(seeds))
        for hit, offset in zip(hits, jitter):
            x = hit if hit is not None else not_reached_x
            ax_steps.scatter([x], [row + offset], s=34, marker=MARKERS[algo],
                             facecolors=COLORS[algo] if hit is not None else "none",
                             edgecolors=COLORS[algo], linewidths=1.4, zorder=3)
        if len(reached) > len(seeds) / 2:
            median = np.median(reached)
            ax_steps.plot([median, median], [row - 0.3, row + 0.3], color=INK,
                          linewidth=1.8, zorder=4)
    ax_steps.axvspan(6e5, 1.3e6, color=GRID, alpha=0.5, zorder=0)
    ax_steps.text(8.9e5, -0.62, "not\nreached", ha="center", va="center", fontsize=8, color=INK_2)
    ax_steps.set_xscale("log")
    ax_steps.set_xlim(1.5e3, 1.3e6)
    step_ticks(ax_steps, 4e5)
    ax_steps.set_yticks(range(len(ALGOS)))
    ax_steps.set_yticklabels([LABELS[a] for a in ALGOS], fontsize=10.5, color=INK)
    ax_steps.set_ylim(len(ALGOS) - 0.4, -0.95)
    ax_steps.set_xlabel(f"First eval point with mean return ≥ {THRESHOLD:.0f}  (env steps)",
                        color=INK_2, fontsize=10)
    ax_steps.set_title("Sample efficiency per seed  (| = median)", loc="left",
                       color=INK, fontsize=11.5, pad=44)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- 图 3：固定评估初始状态
def eval_start_angles(eval_seed, count):
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    angles = []
    for i in range(count):
        obs, _ = env.reset(seed=eval_seed + i)
        angles.append(np.degrees(np.arctan2(obs[1], obs[0])))
    env.close()
    return np.asarray(angles)


def plot_eval_states(runs, seeds, out_path, eval_seed=10000):
    count = len(runs[(ALGOS[0], seeds[0])]["final_eval"]["returns"])
    angles = eval_start_angles(eval_seed, count)
    order = np.argsort(np.abs(angles))
    matrix = np.array([[np.median([runs[(a, s)]["final_eval"]["returns"][i] for s in seeds])
                        for a in ALGOS] for i in order])

    cmap = LinearSegmentedColormap.from_list("ret", ["#f2f6fb", "#9ec3ec", "#2a78d6", "#0e3f7a"]).reversed()
    low, high = -300.0, 0.0
    fig, ax = plt.subplots(figsize=(7.4, 8.2), facecolor=SURFACE)
    ax.imshow(np.clip(matrix, low, high), cmap=cmap, vmin=low, vmax=high, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            rgba = cmap((np.clip(matrix[row, col], low, high) - low) / (high - low))
            ax.text(col, row, f"{matrix[row, col]:.0f}".replace("-0", "0"), ha="center", va="center",
                    fontsize=8.5, color=text_color(rgba))
    ax.set_xticks(range(len(ALGOS)))
    ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=10.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(range(count))
    ax.set_yticklabels([f"{abs(angles[i]):4.0f}°  (seed {eval_seed + i})" for i in order],
                       fontsize=8.5, color=INK_2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.axvline(2.5, color=SURFACE, linewidth=5)
    ax.set_ylabel("Start angle from upright  (0° = upright, 180° = hanging down)",
                  color=INK_2, fontsize=9.5)
    ax.set_title("Return on each fixed evaluation start state\n"
                 f"(median over {len(seeds)} seeds, final policy)", loc="left",
                 color=INK, fontsize=11.5, pad=34)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.results)
    runs = load_runs(out_dir / "raw")
    seeds = sorted({seed for _, seed in runs})
    missing = [(a, s) for a in ALGOS for s in seeds if (a, s) not in runs]
    if missing:
        raise SystemExit(f"缺少运行记录：{missing}")
    summary = write_tables(runs, seeds, out_dir)
    plot_learning_curves(runs, seeds, out_dir / "fig1_learning_curves.png")
    plot_final_by_seed(runs, seeds, out_dir / "fig2_final_by_seed.png")
    plot_eval_states(runs, seeds, out_dir / "fig3_eval_states.png")
    print((out_dir / "tables.md").read_text(encoding="utf-8"))
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
