"""聚合 her_ratio 消融（默认 seeds 0-9）。

数据来源：
- ratio=0   ：seed_experiments/her_main 的 noher_s*.json（纯 DDPG）
- ratio=0.8 ：seed_experiments/her_main 的 her_r0.8_future_s*.json
- ratio=0.2 / 0.5 / 1.0 ：本目录 results/raw/ratio*_s*.json

为什么这样聚合（相对第一版的修正）：

1. 每个指标都带不确定性，而不是只给中位数 / IQR。
   n=10 时 IQR 没什么统计意义，所以用：
   - 平均成功率：对 seed 做 bootstrap 的 95% 区间；
   - “成功 seed 比例”：Wilson 95% 区间（二项比例，n 小时比正态近似可靠）。
   区间重叠时不要写“A 比 B 好”。
2. 收敛回合不再丢掉没收敛的 run。
   第一版只对“收敛了的 seed”取中位数，失败的被悄悄排除（幸存者偏差）。
   现在没收敛按 +inf 计入分位数，显示为 “>2000”，并同时给出 k/n。
3. 成功率不只看终点快照。DDPG 会“到过 1.0 又崩”，终点快照把这类 run 记成 0。
   增加 peak（峰值）、last10（最后 10 个评估点均值）、auc（全程平均成功率，
   同时反映速度和稳定性）。
4. 成功率曲线画“均值 + bootstrap 区间”和“达标 seed 比例”。
   0/1 型数据的中位数曲线会退化成阶梯，看不出有多少 seed 失败。

输出：
- ratio_results.csv           每个 (ratio, seed) 一行
- ratio_table.md              汇总表
- ratio_success_curves.png    左：均值 + 95% bootstrap；右：达标 seed 比例
- ratio_final_success.png     左：逐 seed 最终成功率 + 均值区间；右：达标比例 + Wilson 区间
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
HER_MAIN_RAW = EXPERIMENTS / 'seed_experiments' / 'her_main' / 'results' / 'raw'

RATIOS = [0.0, 0.2, 0.5, 0.8, 1.0]
SUCCESS_LEVEL = 0.9
COLORS = {
    0.0: '#7f7f7f', 0.2: '#1f77b4', 0.5: '#2ca02c',
    0.8: '#d62728', 1.0: '#9467bd',
}
BOOT = 5000


def label_of(ratio):
    return '0 (no HER)' if ratio == 0 else '%.1f' % ratio


# ------------------------------------------------------------------ 指标
def first_success(returns):
    hit = np.where(np.asarray(returns) > -49.5)[0]
    return int(hit[0] + 1) if len(hit) else None


def converge_return(returns, window=20, threshold=-10.0):
    r = np.asarray(returns, dtype=float)
    if len(r) < window:
        return None
    ma = np.convolve(r, np.ones(window) / window, mode='valid')
    hit = np.where(ma > threshold)[0]
    return int(hit[0] + window) if len(hit) else None


def converge_eval(history, level=SUCCESS_LEVEL):
    episodes = [e for e, _ in history]
    values = [s for _, s in history]
    for i in range(len(values)):
        if all(v >= level for v in values[i:]):
            return int(episodes[i])
    return None


def auc_success(history):
    """成功率曲线下面积 / 横轴跨度 = 全程平均成功率（梯形积分）。"""
    e = np.array([x for x, _ in history], dtype=float)
    s = np.array([y for _, y in history], dtype=float)
    area = float(np.sum((s[:-1] + s[1:]) / 2 * np.diff(e)))
    return area / float(e[-1] - e[0])


def run_metrics(d):
    h = np.array([s for _, s in d['success_history']], dtype=float)
    return {
        'final': float(d['final_success']),
        'peak': float(h.max()),
        'last10': float(h[-10:].mean()),
        'auc': auc_success(d['success_history']),
        'first_success': first_success(d['returns']),
        'converge_eval': converge_eval(d['success_history']),
        'converge_return': converge_return(d['returns']),
        'wall': float(d['wall_clock_sec']),
    }


# ------------------------------------------------------------------ 统计
def bootstrap_ci(values, stat=np.mean, boot=BOOT, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(boot, len(v)))
    stats = stat(v[idx], axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def censored_quantiles(values):
    """没达到（None）按 +inf 计入分位数，避免幸存者偏差。"""
    arr = np.array([np.inf if v is None else float(v) for v in values])
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25, method='nearest'))
    q3 = float(np.percentile(arr, 75, method='nearest'))
    return med, q1, q3, int(np.isfinite(arr).sum())


def fmt_episode(x, horizon):
    return '>%d' % horizon if not np.isfinite(x) else '%.0f' % x


# ------------------------------------------------------------------ 读取
def run_path(ratio, seed, raw_dir):
    if ratio == 0.0:
        return HER_MAIN_RAW / f'noher_s{seed}.json'
    if ratio == 0.8:
        return HER_MAIN_RAW / f'her_r0.8_future_s{seed}.json'
    return Path(raw_dir) / f'ratio{ratio}_s{seed}.json'


def load_all(raw_dir, seeds):
    runs = {}
    for ratio in RATIOS:
        for seed in seeds:
            path = run_path(ratio, seed, raw_dir)
            if not path.exists():
                print(f'[missing] ratio={ratio} seed={seed}: {path.name}')
                continue
            d = json.loads(path.read_text(encoding='utf-8'))
            d['metrics'] = run_metrics(d)
            runs[(ratio, seed)] = d
    return runs


def group(runs, ratio):
    return [runs[k] for k in sorted(runs) if k[0] == ratio]


# ------------------------------------------------------------------ 输出
def write_csv(runs, out):
    with (out / 'ratio_results.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['ratio', 'seed', 'final_success', 'peak_success',
                    'last10_mean', 'auc', 'first_success', 'converge_eval',
                    'converge_return', 'wall_clock_sec'])
        for (ratio, seed), d in sorted(runs.items()):
            m = d['metrics']
            w.writerow([ratio, seed, m['final'], m['peak'],
                        round(m['last10'], 3), round(m['auc'], 3),
                        m['first_success'], m['converge_eval'],
                        m['converge_return'], round(m['wall'], 1)])


def write_table(runs, out, horizon):
    lines = [
        '| her_ratio | n | 最终成功率均值 [95%% CI] | 最终>=0.9 的 seed（Wilson 95%%） '
        '| 曾到过>=0.9 | 全程平均成功率 AUC [95%% CI] | 收敛回合（回报，未收敛计 >%d） |' % horizon,
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for ratio in RATIOS:
        items = group(runs, ratio)
        if not items:
            continue
        ms = [d['metrics'] for d in items]
        n = len(ms)
        final = [m['final'] for m in ms]
        auc = [m['auc'] for m in ms]
        k = sum(1 for v in final if v >= SUCCESS_LEVEL)
        peak_k = sum(1 for m in ms if m['peak'] >= SUCCESS_LEVEL)
        f_lo, f_hi = bootstrap_ci(final)
        a_lo, a_hi = bootstrap_ci(auc)
        w_lo, w_hi = wilson_ci(k, n)
        med, q1, q3, got = censored_quantiles([m['converge_return'] for m in ms])
        lines.append(
            '| %s | %d | %.2f [%.2f, %.2f] | %d/%d [%.2f, %.2f] | %d/%d '
            '| %.2f [%.2f, %.2f] | %s [%s, %s]（%d/%d 收敛） |' % (
                label_of(ratio), n, np.mean(final), f_lo, f_hi,
                k, n, w_lo, w_hi, peak_k, n,
                np.mean(auc), a_lo, a_hi,
                fmt_episode(med, horizon), fmt_episode(q1, horizon),
                fmt_episode(q3, horizon), got, n))
    table = '\n'.join(lines)
    (out / 'ratio_table.md').write_text(table + '\n', encoding='utf-8')
    print(table)


def plot_curves(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)
    rng = np.random.default_rng(0)
    for ratio in RATIOS:
        items = group(runs, ratio)
        if not items:
            continue
        episodes = np.array([e for e, _ in items[0]['success_history']])
        mat = np.array([[s for _, s in d['success_history']] for d in items])
        n = len(mat)
        idx = rng.integers(0, n, size=(2000, n))
        boots = mat[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        color = COLORS[ratio]
        axes[0].fill_between(episodes, lo, hi, color=color, alpha=0.14)
        axes[0].plot(episodes, mat.mean(axis=0), color=color, linewidth=2.0,
                     label='ratio=%s (n=%d)' % (label_of(ratio), n))
        axes[1].plot(episodes, (mat >= SUCCESS_LEVEL).mean(axis=0),
                     color=color, linewidth=2.0, label=label_of(ratio))
    axes[0].set_title('Mean success rate (95% bootstrap over seeds)')
    axes[0].set_ylabel('Success rate on 20 fixed goals')
    axes[1].set_title('Fraction of seeds with success >= %.1f' % SUCCESS_LEVEL)
    axes[1].set_ylabel('Fraction of seeds')
    for ax in axes:
        ax.set_xlabel('Episode')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.text(0.01, 0.005,
             'ratio=0 is plain DDPG; ratio=1.0 relabels every sampled transition. '
             'Median curves are not used: 0/1-like data collapses to steps.',
             fontsize=7, color='0.4')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / 'ratio_success_curves.png', bbox_inches='tight')
    plt.close(fig)


def plot_final(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    rng = np.random.default_rng(1)
    for pos, ratio in enumerate(RATIOS, start=1):
        items = group(runs, ratio)
        if not items:
            continue
        ms = [d['metrics'] for d in items]
        final = np.array([m['final'] for m in ms])
        color = COLORS[ratio]
        jitter = rng.uniform(-0.16, 0.16, size=len(final))
        axes[0].scatter(pos + jitter, final, s=26, color=color,
                        edgecolor='white', linewidth=0.6, alpha=0.9, zorder=3)
        lo, hi = bootstrap_ci(final)
        axes[0].errorbar(pos + 0.28, final.mean(),
                         yerr=[[final.mean() - lo], [hi - final.mean()]],
                         fmt='D', color='black', markersize=5, capsize=4,
                         zorder=4)
        k = int((final >= SUCCESS_LEVEL).sum())
        n = len(final)
        w_lo, w_hi = wilson_ci(k, n)
        axes[1].bar(pos, k / n, color=color, alpha=0.55, width=0.6)
        axes[1].errorbar(pos, k / n, yerr=[[k / n - w_lo], [w_hi - k / n]],
                         fmt='none', ecolor='black', capsize=5)
        axes[1].text(pos, min(1.0, w_hi) + 0.025, '%d/%d' % (k, n),
                     ha='center', fontsize=9)
    ticks = list(range(1, len(RATIOS) + 1))
    labels = ['0\n(no HER)', '0.2', '0.5', '0.8', '1.0']
    axes[0].set_title('Final success per seed (diamond: mean, 95% bootstrap)')
    axes[0].set_ylabel('Final success rate (20 goals)')
    axes[1].set_title('Seeds reaching >= %.1f at the end (Wilson 95%%)' % SUCCESS_LEVEL)
    axes[1].set_ylabel('Fraction of seeds')
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel('HER ratio')
        ax.set_ylim(-0.05, 1.15)
        ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / 'ratio_final_success.png', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default=str(HERE / 'results' / 'raw'))
    parser.add_argument('--out', default=str(HERE / 'results'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_all(args.raw, args.seeds)
    if not runs:
        raise SystemExit('没有可用结果')
    horizon = max(len(d['returns']) for d in runs.values())

    write_csv(runs, out)
    write_table(runs, out, horizon)
    plot_curves(runs, out)
    plot_final(runs, out)
    print('saved', out)


if __name__ == '__main__':
    main()
