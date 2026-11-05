"""聚合 SAC 超参单因素扫描，并与 1M 基线的前 100k 步对比。

指标：评估网格 60k/80k/100k 三点的均值（每点 5 回合），降低单点评估噪声。
统计：同一个 seed 的配置 vs 基线做**配对差**（同 seed 同初始网络，前 10k 步不更新，
理论上起点相同），再对 3 个差值做配对 t 检验。n=3、df=2，p 值只作量级参考，
6 个配置同时比较时未做多重比较校正——表里的 p 越小越“值得继续看”，不是“已证明”。
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
BASE_RAW = HERE.parents[1] / 'seed_experiments' / 'halfcheetah_sac' / 'results' / 'raw'
MAX_STEP = 100_000
LATE = (60_000, 80_000, 100_000)
EARLY = (20_000, 40_000)
SEEDS = (0, 1, 2)
ORDER = ['baseline', 'lr1e-4', 'lr1e-3', 'batch64', 'batch1024', 'gamma0.95', 'alpha1.0']
LABEL = {
    'baseline': 'baseline (lr3e-4, b256, g.99, a0.01)',
    'lr1e-4': 'lr 1e-4', 'lr1e-3': 'lr 1e-3',
    'batch64': 'batch 64', 'batch1024': 'batch 1024',
    'gamma0.95': 'gamma 0.95', 'alpha1.0': 'init alpha 1.0',
}


def curve(d):
    """截取评估序列的前 100k 步。

    基线每 20k 步评估一次，截断后正好是 20k/40k/60k/80k/100k 五个点，
    与扫描 run 的评估网格逐点对齐，可以直接比较。
    """
    s = np.array(d['eval']['steps']); r = np.array(d['eval']['returns'])
    keep = s <= MAX_STEP
    return s[keep], r[keep]


def load(name):
    out = {}
    for seed in SEEDS:
        path = (BASE_RAW / f'sac_s{seed}.json') if name == 'baseline' \
            else HERE / 'results' / 'raw' / f'{name}_s{seed}.json'
        if path.exists():
            out[seed] = curve(json.loads(path.read_text(encoding='utf-8')))
    return out


def window_mean(runs, steps):
    """每个 seed 在指定评估点上的平均回报，返回按 seed 排序的数组。"""
    return np.array([r[np.isin(s, steps)].mean() for _, (s, r) in sorted(runs.items())])


def plot_alpha_diagnostic(out):
    """若存在 diagnose_alpha.py 的输出，画 alpha / 熵 / |tanh(mu)| / Q 的前 30k 步轨迹。"""
    paths = {a: HERE / 'results' / 'diagnostics' / f'alpha_trace_a{a}_s0.json'
             for a in ('0.01', '1.0')}
    if not all(p.exists() for p in paths.values()):
        return
    traces = {a: json.loads(p.read_text(encoding='utf-8')) for a, p in paths.items()}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), dpi=160)
    panels = [('alpha', 'alpha (log scale)'), ('entropy', 'policy entropy (target -6)'),
              ('mean_abs_mu', 'mean |tanh(mu)|'), ('q_mean', 'mean min-Q')]
    for ax, (key, title) in zip(axes, panels):
        for a, color in (('0.01', 'black'), ('1.0', '#e377c2')):
            xs = [t['step'] for t in traces[a]]
            ax.plot(xs, [t[key] for t in traces[a]], color=color, linewidth=2,
                    label='init alpha %s' % a)
        if key == 'alpha':
            ax.set_yscale('log')
        if key == 'entropy':
            ax.axhline(-6, color='gray', linestyle='--', linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Environment steps')
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Alpha diagnostic, seed 0 (updates start at 10k)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'alpha_diagnostic.png', bbox_inches='tight')
    plt.close(fig)


def main():
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    groups = {n: load(n) for n in ORDER}
    base_late = window_mean(groups['baseline'], LATE)
    base_early = window_mean(groups['baseline'], EARLY)

    rows = []
    lines = [
        '| 配置 | 60k/80k/100k 均值 ± std(ddof=1) | 逐 seed（s0/s1/s2） | 相对基线 | 配对差均值 | t (df=2) | p（未校正） | 同号 seed | 20k+40k 均值 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in ORDER:
        runs = groups[n]
        if not runs:
            continue
        late = window_mean(runs, LATE)
        early = window_mean(runs, EARLY)
        per_seed = '/'.join('%.0f' % x for x in late)
        if n == 'baseline':
            lines.append('| %s | %.0f ± %.0f | %s | | | | | | %.0f |' % (
                LABEL[n], late.mean(), late.std(ddof=1), per_seed, early.mean()))
            rows.append([n, len(runs), round(late.mean(), 1), round(late.std(ddof=1), 1),
                         '', '', '', '', round(early.mean(), 1)])
            continue
        diff = late - base_late
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        p = 2 * stats.t.sf(abs(t), df=len(diff) - 1)
        same = int(max((diff > 0).sum(), (diff < 0).sum()))
        rel = 100 * diff.mean() / abs(base_late.mean())
        lines.append('| %s | %.0f ± %.0f | %s | %+.0f%% | %+.0f | %.2f | %.3f | %d/3 | %.0f |' % (
            LABEL[n], late.mean(), late.std(ddof=1), per_seed, rel,
            diff.mean(), t, p, same, early.mean()))
        rows.append([n, len(runs), round(late.mean(), 1), round(late.std(ddof=1), 1),
                     round(rel, 1), round(diff.mean(), 1), round(t, 2), round(p, 3),
                     round(early.mean(), 1)])
    table = '\n'.join(lines)
    (out / 'sweep_table.md').write_text(table + '\n', encoding='utf-8')
    with (out / 'sweep_results.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['config', 'seeds', 'late_mean', 'late_std_ddof1', 'rel_pct',
                    'paired_diff_mean', 't', 'p_uncorrected', 'early_mean'])
        w.writerows(rows)
    print(table)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    cmap = plt.get_cmap('tab10')
    for i, n in enumerate(ORDER):
        runs = groups[n]
        if not runs:
            continue
        steps = next(iter(runs.values()))[0]
        mat = np.array([r for _, r in runs.values()])
        color = 'black' if n == 'baseline' else cmap(i)
        ax.plot(steps, mat.mean(axis=0), color=color,
                linewidth=2.6 if n == 'baseline' else 1.8, label=LABEL[n])
        if len(mat) > 1:
            ax.fill_between(steps, mat.min(axis=0), mat.max(axis=0), color=color, alpha=0.07)
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Deterministic eval return (5 episodes)')
    ax.set_title('SAC HalfCheetah-v5 hyperparameters (mean, min–max over 3 seeds)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'sweep_curves.png', bbox_inches='tight')
    plt.close(fig)
    plot_alpha_diagnostic(out)
    print('saved', out)


if __name__ == '__main__':
    main()
