"""聚合 results/raw 下的单次实验，输出 CSV、summary.md 和图。

输入：run_one.py 写出的若干 JSON（每个文件 = 一次 seed × 一组设置）。
输出：
    runs.csv            每次运行一行，便于用 pandas / Excel 再分析；
    summary_table.md    按设置分组的中位数 [IQR]，直接贴进 README；
    success_curves.png  每组一条中位数曲线，细线是单 seed。

指标（都按回合数，不按步数；一回合最多 50 步）：
- first_success：训练中第一次到达目标（回报 > -50）的回合；
- converge_eval：固定评估集成功率 >= 0.9 且此后不再跌破 0.9 的最早回合；
- converge_return：训练回报 20 回合滑动平均首次 > -10 的回合
  （成功回报约 -4，失败恒为 -50；这个指标不依赖评估，可与教程原版直接比较）；
- final_success：训练结束后在固定 20 个目标上的确定性成功率。

为什么用中位数 / IQR 而不是均值：
HER 在这套设置下有 2/10 个 seed 会崩溃（见 经验总结.md），
均值会被少数极端值拖偏；中位数 + 四分位距 + "≥0.9 的 seed 数"
三件一起看，才不会被一个好看的平均数骗过去。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

# 用无显示后端：这个脚本可能跑在服务器上，只需要存图不需要弹窗。
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def converge_return(returns, window=20, threshold=-10.0):
    """训练回报首次稳定超过 threshold 的回合；从未达到返回 None。

    np.convolve(..., mode='valid') 的第 i 个值对应窗口
    returns[i : i+window]，即**第 i+window 回合结束**时的均值（1-based），
    所以返回 hit[0] + window 而不是 hit[0]。
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < window:
        return None
    ma = np.convolve(r, np.ones(window) / window, mode='valid')
    hit = np.where(ma > threshold)[0]
    return int(hit[0] + window) if len(hit) else None


def first_success(returns):
    """训练中第一次成功到达目标的回合。

    成功回报是 −(到达步数)，一定大于 −50；失败恰好是 −50。
    阈值取 −49.5 而不是 −50，是为了避开浮点误差和"最后一步刚好到达
    但回报仍按 50 步计"之类的边界情况。
    """
    hit = np.where(np.asarray(returns) > -49.5)[0]
    return int(hit[0] + 1) if len(hit) else None


def converge_eval(history, level=0.9):
    """评估成功率从某一回合起**再也没跌破** level 的最早回合。

    从前往后找第一个满足"从 i 到末尾全部 >= level"的位置。
    这比"首次 >= 0.9"严格：DDPG 训练不稳定，首次达标后常常又掉下去，
    用首次达标会把"闪一下就崩"的 run 误判成收敛。
    history 只有几十个点，O(n²) 无所谓。
    """
    episodes = [e for e, _ in history]
    values = [s for _, s in history]
    for i in range(len(values)):
        if all(v >= level for v in values[i:]):
            return int(episodes[i])
    return None


def load(raw_dir):
    """读入 raw 目录下所有 JSON，并算好每个 run 的派生指标。"""
    runs = []
    for path in sorted(Path(raw_dir).glob('*.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        # 文件名当 run 的名字，CSV 里查错时方便定位是哪一个 JSON。
        d['name'] = path.stem
        d['first_success'] = first_success(d['returns'])
        d['converge_eval'] = converge_eval(d['success_history'])
        d['converge_return'] = converge_return(d['returns'])
        runs.append(d)
    return runs


def group_key(d):
    """把一次 run 映射到"实验设置"这一列：无 HER 合并成一组，HER 按比例+策略分。"""
    if not d['her']:
        return 'no HER'
    return 'HER ratio=%.1f %s' % (d['ratio'], d['strategy'])


def median_iqr(values):
    """去掉 None 后算 (中位数, 25% 分位, 75% 分位)；空列表返回三个 None。"""
    values = [v for v in values if v is not None]
    if not values:
        return None, None, None
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return float(med), float(q1), float(q3)


def fmt(values, n_total):
    """把一个指标的一列 seed 格式化成 "中位数 [Q1, Q3]"，并标注缺失数。

    例："never (0/10)" 表示 10 个 seed 全部没达到；
        "408 [317, 426] (8/10)" 表示 10 个里只有 8 个达到，另外 2 个被排除在
        中位数之外——这个后缀很重要，否则中位数会掩盖失败。
    """
    med, q1, q3 = median_iqr(values)
    got = len([v for v in values if v is not None])
    if med is None:
        return 'never (0/%d)' % n_total
    tag = '' if got == n_total else ' (%d/%d)' % (got, n_total)
    return '%.0f [%.0f, %.0f]%s' % (med, q1, q3, tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default=str(HERE / 'results' / 'raw'))
    parser.add_argument('--out', default=str(HERE / 'results'))
    parser.add_argument('--prefix', default='',
                        help='输出文件名前缀，避免多批实验互相覆盖')
    args = parser.parse_args()

    out = Path(args.out)
    runs = load(args.raw)
    # 按设置分组；dict 保持插入顺序，所以图的图例顺序 = 首次出现的顺序。
    groups = {}
    for d in runs:
        groups.setdefault(group_key(d), []).append(d)

    # ------------------------------------------------------------- CSV
    # 一次 run 一行，保留原始字段，方便之后用 pandas 做别的统计。
    with (out / (args.prefix + 'runs.csv')).open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'seed', 'first_success', 'converge_eval',
                    'converge_return', 'final_success', 'wall_clock_sec'])
        for key, items in groups.items():
            for d in sorted(items, key=lambda x: x['seed']):
                w.writerow([key, d['seed'], d['first_success'], d['converge_eval'],
                            d['converge_return'], d['final_success'],
                            round(d['wall_clock_sec'], 1)])

    # ------------------------------------------------------- Markdown 汇总
    # 每个指标列：中位数 [Q1, Q3]；若有机 run 缺失，会带 (k/n) 后缀。
    lines = ['| 设置 | seeds | 首次成功回合 | 收敛回合（评估） | 收敛回合（回报） | 最终成功率 | ≥0.9 seeds | 耗时 s |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, items in groups.items():
        n = len(items)
        fs = [d['first_success'] for d in items]
        ce = [d['converge_eval'] for d in items]
        cr = [d['converge_return'] for d in items]
        fin = [d['final_success'] for d in items]
        wall = [d['wall_clock_sec'] for d in items]
        # "成功"的判定线：最终成功率 >= 0.9。直接报个数，不藏在均值里。
        good = int(sum(1 for v in fin if v >= 0.9))
        lines.append('| %s | %d | %s | %s | %s | %.2f [%.2f, %.2f] | %d/%d | %.0f |' % (
            key, n, fmt(fs, n), fmt(ce, n), fmt(cr, n),
            np.median(fin), np.percentile(fin, 25), np.percentile(fin, 75),
            good, n, np.median(wall)))
    table = '\n'.join(lines)
    (out / (args.prefix + 'summary_table.md')).write_text(table + '\n', encoding='utf-8')
    print(table)

    # ---------------------------------------------------------------- 图
    # 图：每组成功率曲线，细线 = 单 seed，粗线 = 中位数。
    # 细线不能省：中位数会把"2/10 崩溃"这种事实抹平，细线能直接看到崩溃形态。
    palette = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for color, (key, items) in zip(palette, groups.items()):
        # 假设同一组内所有 run 的评估回合点一致（eval_interval 相同）。
        episodes = [e for e, _ in items[0]['success_history']]
        # 每一行是一个 seed 的成功率序列。
        mat = np.array([[s for _, s in d['success_history']] for d in items])
        for row in mat:
            ax.plot(episodes, row, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(episodes, np.median(mat, axis=0), color=color, linewidth=2.4,
                label='%s (n=%d)' % (key, len(items)))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success rate on 20 fixed goals')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('GridWorld: DDPG with / without HER')
    fig.text(0.01, 0.005, 'Thin = single seed, thick = median across seeds. '
             'Deterministic evaluation on fixed goals.', fontsize=7, color='0.4')
    # rect 给底部脚注留 3% 的空间，避免 tight_layout 把它裁掉。
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / (args.prefix + 'success_curves.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
