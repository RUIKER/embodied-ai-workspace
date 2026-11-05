"""聚合 HalfCheetah SAC 的 3 个 seed：学习曲线、最终 return 表和图。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLOR = '#1f77b4'


def load(raw_dir):
    runs = {}
    for path in sorted(Path(raw_dir).glob('sac_s*.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        runs[d['seed']] = d
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default=str(HERE / 'results' / 'raw'))
    parser.add_argument('--out', default=str(HERE / 'results'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load(args.raw)
    if not runs:
        raise SystemExit('没有结果')

    # 逐 seed CSV
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'steps', 'final_mean', 'final_median',
                    'final_std', 'final_best', 'wall_clock_sec'])
        for seed, d in sorted(runs.items()):
            f = d['final_eval']
            w.writerow([seed, d['steps'], f['mean'], f['median'],
                        f['std'], f['best'], round(d['wall_clock_sec'], 1)])

    # 学习曲线：每 20k 一个评估点；3 条 seed 曲线画淡色底，中位数画粗线。
    # 用中位数而不是均值，是因为 3 个 seed 里一旦有一个崩掉，均值会被拽偏。
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    steps = np.array(next(iter(runs.values()))['eval']['steps'])
    mat = np.array([d['eval']['returns'] for d in runs.values()])
    for row in mat:
        ax.plot(steps, row, color=COLOR, alpha=0.22, linewidth=0.9)
    if mat.shape[0] > 1:
        # 每个 step 位置跨 seed 取四分位；n=3 时 IQR 只是“最小到最大”的近似，
        # 不要当成统计置信区间。
        q1, med, q3 = np.percentile(mat, [25, 50, 75], axis=0)
        ax.fill_between(steps, q1, q3, color=COLOR, alpha=0.15)
        ax.plot(steps, med, color=COLOR, linewidth=2.4,
                label='median (n=%d)' % mat.shape[0])
    else:
        ax.plot(steps, mat[0], color=COLOR, linewidth=2.4, label='seed 0')
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Deterministic eval return (5 episodes)')
    ax.set_title('SAC on HalfCheetah-v5')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png', bbox_inches='tight')
    plt.close(fig)

    # 最终评估：20 回合的均值，每个点一个 seed。
    # 终点数值比曲线更依赖评估回合数，报告时要和 SB3 的评估协议对齐。
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    final = [d['final_eval']['mean'] for _, d in sorted(runs.items())]
    seeds = [s for s, _ in sorted(runs.items())]
    ax.scatter(seeds, final, color=COLOR, s=40, zorder=3)
    ax.plot(seeds, final, color=COLOR, alpha=0.5)
    ax.set_xticks(seeds)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Final eval return (20 episodes)')
    ax.set_title('HalfCheetah SAC: final return')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / 'final_eval.png', bbox_inches='tight')
    plt.close(fig)

    # Markdown 摘要
    lines = [
        '# HalfCheetah SAC（3 seeds）',
        '',
        '| seed | steps | final mean | final median | final best | wall (s) |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for seed, d in sorted(runs.items()):
        f = d['final_eval']
        lines.append('| %d | %d | %.1f | %.1f | %.1f | %.0f |' % (
            seed, d['steps'], f['mean'], f['median'], f['best'],
            d['wall_clock_sec']))
    vals = [d['final_eval']['mean'] for d in runs.values()]
    lines += ['', '最终 return 均值：%.1f' % np.mean(vals),
              '中位数：%.1f' % np.median(vals)]
    (out / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
