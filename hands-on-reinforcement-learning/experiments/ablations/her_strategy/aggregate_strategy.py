"""聚合 HER 目标采样策略消融：future / final / episode。

future 复用 seed_experiments/her_main 的 ratio=0.8 主实验（10 seeds）；
final / episode 由本目录 results/raw 提供。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
FUTURE_RAW = EXPERIMENTS / 'seed_experiments' / 'her_main' / 'results' / 'raw'

STRATEGIES = [('future', 'future'), ('final', 'final'), ('episode', 'episode')]
COLORS = {'future': '#d62728', 'final': '#1f77b4', 'episode': '#2ca02c'}
SEEDS = list(range(10))
LEVEL = 0.9


def converge_return(returns, window=20, threshold=-10.0):
    r = np.asarray(returns, dtype=float)
    if len(r) < window:
        return None
    ma = np.convolve(r, np.ones(window) / window, mode='valid')
    hit = np.where(ma > threshold)[0]
    return int(hit[0] + window) if len(hit) else None


def auc(history):
    e = np.array([x for x, _ in history], dtype=float)
    s = np.array([y for _, y in history], dtype=float)
    return float(np.sum((s[:-1] + s[1:]) / 2 * np.diff(e)) / (e[-1] - e[0]))


def load(strategy, raw_dir):
    runs = {}
    for seed in SEEDS:
        if strategy == 'future':
            path = FUTURE_RAW / f'her_r0.8_future_s{seed}.json'
        else:
            path = Path(raw_dir) / f'{strategy}_s{seed}.json'
        if path.exists():
            d = json.loads(path.read_text(encoding='utf-8'))
            d['converge'] = converge_return(d['returns'])
            d['auc'] = auc(d['success_history'])
            runs[seed] = d
    return runs


def fmt_ep(x):
    return '>2000' if x is None else '%.0f' % x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default=str(HERE / 'results' / 'raw'))
    parser.add_argument('--out', default=str(HERE / 'results'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = {s: load(s, args.raw) for s, _ in STRATEGIES}

    with (out / 'strategy_results.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['strategy', 'seed', 'final_success', 'peak_success',
                    'auc', 'converge_return'])
        for name, runs in groups.items():
            for seed, d in sorted(runs.items()):
                h = [x for _, x in d['success_history']]
                w.writerow([name, seed, d['final_success'], max(h),
                            round(d['auc'], 3), d['converge']])

    lines = [
        '| 策略 | seeds | 最终成功率均值 [95% bootstrap] | >=0.9 seeds | 全程平均成功率 AUC | 收敛回合（回报，未收敛计 >2000） |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name, runs in groups.items():
        if not runs:
            continue
        final = np.array([d['final_success'] for d in runs.values()])
        aucs = np.array([d['auc'] for d in runs.values()])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(final), size=(3000, len(final)))
        lo, hi = np.percentile(final[idx].mean(axis=1), [2.5, 97.5])
        good = int((final >= LEVEL).sum())
        conv = [d['converge'] for d in runs.values()]
        finite = [c for c in conv if c is not None]
        lines.append('| %s | %d | %.2f [%.2f, %.2f] | %d/%d | %.2f | %s（%d/%d） |' % (
            name, len(final), final.mean(), lo, hi, good, len(final),
            aucs.mean(),
            fmt_ep(np.median(finite) if finite else None), len(finite), len(final)))
    table = '\n'.join(lines)
    (out / 'strategy_table.md').write_text(table + '\n', encoding='utf-8')
    print(table)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    for name, runs in groups.items():
        if not runs:
            continue
        episodes = np.array([e for e, _ in next(iter(runs.values()))['success_history']])
        mat = np.array([[s for _, s in d['success_history']] for d in runs.values()])
        color = COLORS[name]
        q1, med, q3 = np.percentile(mat, [25, 50, 75], axis=0)
        axes[0].fill_between(episodes, q1, q3, color=color, alpha=0.13)
        axes[0].plot(episodes, med, color=color, linewidth=2,
                     label='%s (n=%d)' % (name, len(mat)))
        axes[1].plot(episodes, (mat >= LEVEL).mean(axis=0),
                     color=color, linewidth=2, label=name)
    axes[0].set_title('Success rate (median + IQR)')
    axes[0].set_ylabel('Success rate on 20 fixed goals')
    axes[1].set_title('Fraction of seeds >= %.1f' % LEVEL)
    axes[1].set_ylabel('Fraction of seeds')
    for ax in axes:
        ax.set_xlabel('Episode')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'strategy_success_curves.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    data = [[d['final_success'] for d in groups[name].values()] for name, _ in STRATEGIES]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, (name, _) in zip(bp['boxes'], STRATEGIES):
        patch.set_facecolor(COLORS[name]); patch.set_alpha(0.3)
    for i, (name, _) in enumerate(STRATEGIES, start=1):
        vals = [d['final_success'] for d in groups[name].values()]
        ax.scatter(i + np.linspace(-.1, .1, len(vals)), vals,
                   color=COLORS[name], s=24, zorder=3)
    ax.set_xticks(range(1, len(STRATEGIES) + 1))
    ax.set_xticklabels([n for n, _ in STRATEGIES])
    ax.set_ylabel('Final success rate')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis='y', alpha=0.25)
    ax.set_title('HER goal strategy: final success (10 seeds)')
    fig.tight_layout()
    fig.savefig(out / 'strategy_final_success.png', bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
