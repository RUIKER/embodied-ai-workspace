"""SAC 超参灵敏度网格：调度脚本和聚合脚本共用的单一数据源。

每个超参在基线附近取 3–5 个点（对数间隔），每个点 = 一个配置名 + 命令行参数。
`name` 同时是 results/raw/{name}_s{seed}.json 的文件名前缀，已有的 6 个配置
（lr1e-4、lr1e-3、batch64、batch1024、gamma0.95、alpha1.0）沿用原名，不重跑。

基线（value == base）不单独跑，直接取 seed_experiments/halfcheetah_sac 的 1M 结果在
前 100k 步（同代码、同 seed、同评估网格）。

tier 决定调度顺序：先 1 后 2 后 3；同一 tier 内按 seed 外层循环，
所以任何时刻中断，已完成的都是“整齐的前 k 个 seed”。
"""

import sys

# (value, name, extra_args) —— value 用于画图的横轴；value == base 的点是基线。
GRID = [
    dict(key='lr', title='Learning rate (actor = critic)', base=3e-4, log=True, tier=1,
         points=[(1e-4, 'lr1e-4', '--actor-lr 1e-4 --critic-lr 1e-4'),
                 (3e-4, None, ''),
                 (6e-4, 'lr6e-4', '--actor-lr 6e-4 --critic-lr 6e-4'),
                 (1e-3, 'lr1e-3', '--actor-lr 1e-3 --critic-lr 1e-3'),
                 (3e-3, 'lr3e-3', '--actor-lr 3e-3 --critic-lr 3e-3')]),
    dict(key='gamma', title='Discount gamma', base=0.99, log=False, tier=1,
         points=[(0.9, 'gamma0.9', '--gamma 0.9'),
                 (0.95, 'gamma0.95', '--gamma 0.95'),
                 (0.99, None, ''),
                 (0.995, 'gamma0.995', '--gamma 0.995'),
                 (0.998, 'gamma0.998', '--gamma 0.998')]),
    dict(key='tau', title='Target smoothing tau', base=0.005, log=True, tier=1,
         points=[(0.001, 'tau0.001', '--tau 0.001'),
                 (0.005, None, ''),
                 (0.02, 'tau0.02', '--tau 0.02'),
                 (0.1, 'tau0.1', '--tau 0.1')]),
    dict(key='target_entropy', title='Target entropy (baseline -6 = -dim)', base=-6.0,
         log=False, tier=1,
         points=[(-12.0, 'tent-12', '--target-entropy=-12'),
                 (-6.0, None, ''),
                 (-3.0, 'tent-3', '--target-entropy=-3'),
                 (0.0, 'tent0', '--target-entropy=0')]),
    dict(key='batch', title='Batch size', base=256, log=True, tier=1,
         points=[(16, 'batch16', '--batch-size 16'),
                 (64, 'batch64', '--batch-size 64'),
                 (256, None, ''),
                 (1024, 'batch1024', '--batch-size 1024'),
                 (4096, 'batch4096', '--batch-size 4096')]),
    dict(key='utd', title='Updates per env step (UTD)', base=1, log=True, tier=1,
         points=[(1, None, ''),
                 (2, 'utd2', '--utd 2')]),
    dict(key='init_alpha', title='Initial alpha', base=0.01, log=True, tier=2,
         points=[(0.001, 'alpha0.001', '--init-alpha 0.001'),
                 (0.01, None, ''),
                 (0.1, 'alpha0.1', '--init-alpha 0.1'),
                 (1.0, 'alpha1.0', '--init-alpha 1.0')]),
    dict(key='alpha_lr', title='Alpha learning rate', base=3e-4, log=True, tier=2,
         points=[(3e-5, 'alr3e-5', '--alpha-lr 3e-5'),
                 (3e-4, None, ''),
                 (3e-3, 'alr3e-3', '--alpha-lr 3e-3')]),
    dict(key='hidden', title='Hidden width', base=256, log=True, tier=2,
         points=[(64, 'hid64', '--hidden-dim 64'),
                 (256, None, ''),
                 (512, 'hid512', '--hidden-dim 512')]),
    dict(key='reward_scale', title='Reward scale', base=1.0, log=True, tier=2,
         points=[(0.2, 'rs0.2', '--reward-scale 0.2'),
                 (1.0, None, ''),
                 (5.0, 'rs5', '--reward-scale 5')]),
    # UTD=4 的单个 run 约是基线的 4 倍时长，放到最后。
    dict(key='utd4', title='UTD = 4 (heavy)', base=None, log=True, tier=3,
         points=[(4, 'utd4', '--utd 4')]),
]


def configs():
    """所有需要跑的 (tier, name, extra_args)，已去重（保持顺序）。"""
    seen, out = set(), []
    for group in GRID:
        for _, name, extra in group['points']:
            if name and name not in seen:
                seen.add(name)
                out.append((group['tier'], name, extra))
    return out


if __name__ == '__main__':
    # 供 run_sensitivity.sh 读取：每行 "tier|name|args"
    for tier, name, extra in configs():
        sys.stdout.write('%d|%s|%s\n' % (tier, name, extra))
