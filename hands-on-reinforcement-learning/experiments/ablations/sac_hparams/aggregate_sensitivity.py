"""SAC 超参灵敏度图：把 grid.py 里所有配置聚合成“灵敏度地图”。

产出（results/）：
    sensitivity_map.png       每个超参一个小图：横轴=超参取值，纵轴=后期回报（60/80/100k 均值）
    sensitivity_early.png     同上，指标换成早期速度（20/40k 均值）
    sensitivity_tornado.png   每个超参相对基线的最差/最好变化，灰带 = 基线跨 seed 噪声
    sensitivity_table.md/csv  逐配置：均值、相对基线、均值差、t、p、Holm 校正 p、同号 seed 数
    sensitivity_summary.md    逐超参：最差/最好变化、噪声倍数、粗分类

读图规则：
    - 圆点是单个 seed，线是 3 个 seed 的均值，灰带是基线 ± 1 个**池化**跨 seed 标准差；
    - ★ = 能站得住的结论：与基线的均值差用池化方差做 t 检验（所有配置共用同一个 seed
      噪声估计，df = Σ(n_i-1)），再对全部比较做 Holm 校正后 p < 0.05；
    - ◇ = 只有迹象：3/3 seed 同号且 |均值差| > 2 倍池化 std，但没过校正检验；
      面试里只能说“有迹象，没过校正”，不能当结论；
    - ✕ = 至少一个 seed 发散；n<3 的点是还没跑完的，只画不检验。
为什么用池化方差：基线只有 3 个 seed，自估 std 自由度只有 2，既不稳也偏低（基线 9.7%，
全部配置池化后约 13.5%）；而逐配置配对 t 检验对 ~29 个比较做多重校正几乎不可能通过，
模拟还显示旧的“3/3 同号且 >2σ”规则在纯噪声下平均误报约 2.7 个 ★。
局限：假设各配置噪声大致相同，实际不成立（gamma0.95 的 std 约 1046，lr6e-4 约 115），
所以 ★ 的 p 值是近似，结论宜保守。分类只决定阅读顺序，不是统计结论。
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from grid import GRID

HERE = Path(__file__).resolve().parent
BASE_RAW = HERE.parents[1] / 'seed_experiments' / 'halfcheetah_sac' / 'results' / 'raw'
RAW = HERE / 'results' / 'raw'
OUT = HERE / 'results'
SEEDS = (0, 1, 2)
MAX_STEP = 100_000
LATE = (60_000, 80_000, 100_000)
EARLY = (20_000, 40_000)


def nanmean(x):
    """忽略 NaN 的均值；全是 NaN（发散）时返回 NaN，不触发 numpy 警告。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


def load_run(path):
    """读一个 run，返回 {'late','early','diverged'}；文件不存在返回 None。"""
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    s = np.array(d['eval']['steps'])
    r = np.array(d['eval']['returns'], dtype=float)
    keep = s <= MAX_STEP
    s, r = s[keep], r[keep]
    return {
        'late': nanmean(r[np.isin(s, LATE)]),
        'early': nanmean(r[np.isin(s, EARLY)]),
        'diverged': bool(d.get('diverged_at_step') is not None or np.isnan(r).any()),
    }


def load_config(name):
    """name=None 表示基线（1M run 的前 100k 步）。返回 {seed: run}。"""
    out = {}
    for seed in SEEDS:
        path = (BASE_RAW / f'sac_s{seed}.json') if name is None \
            else RAW / f'{name}_s{seed}.json'
        run = load_run(path)
        if run is not None:
            out[seed] = run
    return out


def holm(pvals):
    """Holm–Bonferroni 校正；输入 {key: p}，输出 {key: p_adj}。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def build_records(groups, base):
    """逐配置统计：均值、与基线的均值差、池化方差 t 检验，再对所有 p 做 Holm 校正。

    返回 (records, pooled_std)：pooled_std 是全部配置内 seed std 的均方根。
    """
    records, pvals = [], {}
    for g in groups:
        for value, name, _ in g['points']:
            runs = base if name is None else load_config(name)
            seeds = sorted(runs)
            late = np.array([runs[s]['late'] for s in seeds])
            early = np.array([runs[s]['early'] for s in seeds])
            rec = dict(key=g['key'], value=value, name=name or 'baseline', seeds=seeds,
                       late=late, early=early,
                       diverged=sum(runs[s]['diverged'] for s in seeds),
                       t=np.nan, p=np.nan, diff=np.nan, same=0)
            records.append(rec)

    # 池化 seed 噪声：各配置样本方差按自由度加权。
    ok = [r for r in records if len(r['seeds']) >= 2 and np.isfinite(r['late']).all()]
    dof = sum(len(r['seeds']) - 1 for r in ok)
    pooled_std = float(np.sqrt(
        sum((len(r['seeds']) - 1) * r['late'].var(ddof=1) for r in ok) / dof))
    base_rec = next(r for r in records if r['name'] == 'baseline')
    bn, bmean = len(base_rec['seeds']), base_rec['late'].mean()

    for rec in records:
        # 只有 3 个 seed 都在、且都没发散时才检验。
        if rec['name'] == 'baseline' or len(rec['seeds']) != 3                 or not np.isfinite(rec['late']).all():
            continue
        rec['diff'] = rec['late'].mean() - bmean
        # 同号 seed 数仍按同 seed 配对统计，只作方向一致性的辅助信息。
        pair = rec['late'] - np.array([base[s]['late'] for s in rec['seeds']])
        rec['same'] = int(max((pair > 0).sum(), (pair < 0).sum()))
        se = pooled_std * np.sqrt(1 / len(rec['seeds']) + 1 / bn)
        rec['t'] = rec['diff'] / se
        rec['p'] = 2 * stats.t.sf(abs(rec['t']), df=dof)
        pvals[rec['name']] = rec['p']

    padj = holm(pvals)
    for r in records:
        r['p_holm'] = padj.get(r['name'], np.nan)
        # ★ 是能当结论讲的；◇ 只能说“有迹象”。
        r['sig'] = bool(np.isfinite(r['p_holm']) and r['p_holm'] < 0.05)
        r['hint'] = bool(not r['sig'] and r['same'] == 3 and np.isfinite(r['diff'])
                         and abs(r['diff']) > 2 * pooled_std)
    return records, pooled_std


def fmt(x, spec):
    return '' if (x == '' or (isinstance(x, float) and np.isnan(x))) else spec % x


def write_tables(records, base_mean):
    header = ['hyper', 'value', 'config', 'n', 'late_mean', 'late_std', 'rel_pct',
              'mean_diff', 't', 'p', 'p_holm', 'same_sign', 'early_mean', 'diverged']
    rows = []
    for r in records:
        lm = nanmean(r['late'])
        std = float(np.nanstd(r['late'], ddof=1)) if len(r['seeds']) > 1 else float('nan')
        rel = '' if r['name'] == 'baseline' else 100 * (lm - base_mean) / abs(base_mean)
        rows.append([r['key'], r['value'], r['name'], len(r['seeds']), lm, std, rel,
                     r['diff'], r['t'], r['p'], r['p_holm'],
                     '' if r['name'] == 'baseline' else '%d/3' % r['same'],
                     nanmean(r['early']), r['diverged']])
    with (OUT / 'sensitivity_results.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow([('' if (isinstance(v, float) and np.isnan(v)) else
                         (round(v, 3) if isinstance(v, float) else v)) for v in row])

    md = ['| 超参 | 取值 | 配置 | n | 后期回报 ± std | 相对基线 | 均值差 | t | p | Holm p | 同号 | 早期回报 | 发散 |',
          '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row, r in zip(rows, records):
        star = ' ★' if r['sig'] else (' ◇' if r['hint'] else '')
        rel = fmt(row[6], '%+.0f%%') + star if row[6] != '' else ''
        md.append('| %s | %g | %s | %d | %s ± %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            row[0], row[1], row[2], row[3], fmt(row[4], '%.0f') or 'NaN',
            fmt(row[5], '%.0f') or '—', rel, fmt(row[7], '%+.0f'), fmt(row[8], '%.2f'),
            fmt(row[9], '%.3f'), fmt(row[10], '%.3f'), row[11],
            fmt(row[12], '%.0f') or 'NaN', '%d/%d' % (row[13], row[3]) if row[13] else '0'))
    (OUT / 'sensitivity_table.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def write_summary(groups, records, base_mean, pooled_std, noise_rel, n_base):
    """逐超参摘要 + 返回 tornado 数据。"""
    lines = ['基线后期回报 %.0f（基线 n=%d）；全部配置池化后的 seed 噪声 std = %.0f（%.1f%%）。' % (
        base_mean, n_base, pooled_std, 100 * noise_rel),
        '★ = 池化 t 检验 + Holm 校正后 p<0.05（能当结论）；'
        '◇ = 3/3 同号且 >2σ 但未过校正（只能说有迹象）。', '',
        '| 超参 | 最差点 | 最好点 | 范围/噪声 | ★ 点数 | ◇ 点数 | 发散 run 数 | 粗分类 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    tornado = []
    for g in groups:
        recs = [r for r in records if r['key'] == g['key'] and r['name'] != 'baseline'
                and np.isfinite(nanmean(r['late']))]
        if not recs:
            continue
        rel = {r['name']: (nanmean(r['late']) - base_mean) / abs(base_mean) for r in recs}
        wname, bname = min(rel, key=rel.get), max(rel, key=rel.get)
        worst, best = rel[wname], rel[bname]
        span = (max(0.0, best) - min(0.0, worst)) / noise_rel
        nsig = sum(r['sig'] for r in recs)
        nhint = sum(r['hint'] for r in recs)
        ndiv = sum(r['diverged'] for r in records if r['key'] == g['key'])
        if nsig:
            cls = '敏感（有 ★，过校正检验）'
        elif nhint:
            cls = '有迹象（◇，未过校正，不能当结论）'
        elif max(abs(worst), abs(best)) < noise_rel:
            cls = '钝感（全部在噪声内）'
        else:
            cls = '待定（有超噪声的点，但方向不一致）'
        lines.append('| %s | %s %+.0f%% | %s %+.0f%% | %.1f× | %d | %d | %d | %s |' % (
            g['key'], wname, 100 * worst, bname, 100 * best, span, nsig, nhint, ndiv, cls))
        tornado.append((g['key'], worst, best, nsig, nhint))
    (OUT / 'sensitivity_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return tornado


def small_multiples(groups, records, metric, base_vals, fname, ylabel, band):
    """每个超参一张小图：圆点=单 seed，线=均值，灰带=基线±1std，竖线=基线取值。"""
    n, cols = len(groups), 5
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.1 * cols, 3.5 * rows_n), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    bmean, bstd = base_vals.mean(), band
    for ax, g in zip(axes, groups):
        recs = [r for r in records if r['key'] == g['key']]
        gamma = g['key'] == 'gamma'
        # gamma 用有效视野 1/(1-gamma) 做横轴（对数），比直接画 gamma 更能看出间隔。
        xs = [1 / (1 - r['value']) if gamma else r['value'] for r in recs]
        order = np.argsort(xs)
        ax.axhspan(bmean - bstd, bmean + bstd, color='gray', alpha=0.18, zorder=0)
        ax.axhline(bmean, color='gray', linewidth=1, linestyle='--', zorder=0)
        means = []
        for i in order:
            r, x = recs[i], xs[i]
            vals = r[metric]
            ax.scatter([x] * len(vals), vals, s=14, color='#1f77b4', alpha=0.45, zorder=2)
            m = nanmean(vals)
            if np.isfinite(m):
                means.append((x, m))
            if r['diverged']:
                ax.scatter([x], [bmean * 0.02], marker='x', color='red', s=50, zorder=4)
            if r['sig']:
                ax.scatter([x], [m], marker='*', s=140, color='#d62728', zorder=5)
            elif r['hint']:
                ax.scatter([x], [m], marker='D', s=36, facecolors='none',
                           edgecolors='#ff7f0e', linewidths=1.6, zorder=5)
        if means:
            mx, my = zip(*means)
            ax.plot(mx, my, color='#1f77b4', linewidth=1.8, zorder=3)
        ax.axvline(1 / (1 - g['base']) if gamma else g['base'],
                   color='black', linewidth=0.8, alpha=0.5)
        if g['log'] or gamma:
            ax.set_xscale('log')
        ticks = sorted(set(xs))
        ax.set_xticks(ticks)
        ax.set_xticklabels([('%g' % (1 - 1 / t) if gamma else '%g' % t) for t in ticks],
                           fontsize=7)
        ax.minorticks_off()
        ax.set_title(g['title'], fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis('off')
    for ax in axes[::cols]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle('SAC HalfCheetah-v5 hyperparameter sensitivity (100k steps, 3 seeds; gray band = '
                 'baseline ± 1 pooled std, star = Holm p<0.05, diamond = 3/3 same sign but not corrected, '
                 'x = diverged)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / fname, bbox_inches='tight')
    plt.close(fig)


def plot_tornado(tornado, noise_rel):
    """深色 = 有 ★（过校正检验）；浅色长条不能当结论。"""
    tornado = sorted(tornado, key=lambda t: min(0, t[1]) - max(0, t[2]))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(tornado) + 1.8), dpi=150)
    for i, (k, lo, hi, nsig, nhint) in enumerate(tornado):
        ax.barh(i, 100 * min(0, lo), color='#d62728' if nsig else '#e8a0a0')
        ax.barh(i, 100 * max(0, hi), color='#2ca02c' if nsig else '#9fd29f')
        ax.text(100 * min(0, lo) - 1, i, '%+.0f%%' % (100 * lo), va='center', ha='right', fontsize=8)
        ax.text(100 * max(0, hi) + 1, i, '%+.0f%%' % (100 * hi), va='center', ha='left', fontsize=8)
    ax.axvspan(-100 * noise_rel, 100 * noise_rel, color='gray', alpha=0.2,
               label='pooled seed noise ±%.0f%%' % (100 * noise_rel))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(tornado)))
    ax.set_yticklabels([t[0] for t in tornado])
    ax.invert_yaxis()
    ax.set_xlabel('Change in late return vs baseline (%): worst / best point in the grid')
    ax.set_title('Sensitivity ranking (deep color = has a Holm-significant point)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / 'sensitivity_tornado.png', bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    base = load_config(None)
    base_late = np.array([base[s]['late'] for s in SEEDS if s in base])
    base_early = np.array([base[s]['early'] for s in SEEDS if s in base])
    base_mean = base_late.mean()

    # utd4 单独成组只是为了调度（最慢，放最后），画图时并入 utd。
    utd4 = next(g for g in GRID if g['key'] == 'utd4')
    groups = []
    for g in GRID:
        if g['key'] == 'utd4':
            continue
        g = dict(g)
        if g['key'] == 'utd':
            g['points'] = g['points'] + utd4['points']
        groups.append(g)

    records, pooled_std = build_records(groups, base)
    noise_rel = pooled_std / abs(base_mean)
    write_tables(records, base_mean)
    tornado = write_summary(groups, records, base_mean, pooled_std, noise_rel,
                            len(base_late))
    small_multiples(groups, records, 'late', base_late, 'sensitivity_map.png',
                    'Eval return, mean of 60k/80k/100k', pooled_std)
    small_multiples(groups, records, 'early', base_early, 'sensitivity_early.png',
                    'Eval return, mean of 20k/40k', pooled_std)
    if tornado:
        plot_tornado(tornado, noise_rel)
    print('saved', OUT)


if __name__ == '__main__':
    main()
