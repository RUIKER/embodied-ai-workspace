# 五算法固定 seed 对比（Pendulum-v1）

DDPG / TD3 / SAC（off-policy）与 PPO / TRPO（on-policy），seeds 0–9，统一评估协议。
结论和分析见 **[`经验总结.md`](经验总结.md)**；本文件只说明怎么跑、产出什么。

## 固定 seed 协议

| 项 | 设置 |
|---|---|
| 环境 | `Pendulum-v1`，每回合 200 步 |
| seeds | `0–9`；seed 决定网络初始化，以及训练中每个回合的 `env.reset(seed + episode)` |
| 评估初始状态 | `EVAL_SEED=10000`，固定 20 个起始状态（10000–10019）；**所有算法、所有 seed、所有评估点完全相同** |
| 评估方式 | 确定性策略（DDPG/TD3 `actor(s)`，SAC `tanh(mu)`，PPO/TRPO `mu`） |
| 评估频率 | off-policy 每 2k 步；on-policy 每 10k 步；每次 20 回合 |
| 预算 | off-policy 40k 步；on-policy 400k 步 |
| 设备 | CPU（网络很小；on-policy 400k 步约 30–70 s。旧版在另一台机器的 GPU 上约 350–420 s，非同机对照） |
| 留出 seed | seeds 0–2 在调 PPO/TRPO 配置时看过，3–9 是留出集；图和表里会标出来 |

训练回报含探索噪声，**只用于画训练曲线，不用于算法间比较**；所有性能数字都来自上面的确定性评估。

## 用法

```bash
cd <repo>/experiments/seed_experiments/five_algorithms
python -I run_all.py          # 50 次运行，已有结果自动跳过，约 5 分钟
python -I analyze.py          # 生成表格和三张图
```

常用选项：`run_all.py --seeds 0 1 2 --algos ppo trpo --force`；
`run_one.py` / `run_one_onpolicy.py` 可单独调试，`--set KEY=VALUE` 覆盖 config 里的超参。

## 产出（`results/`）

| 文件 | 内容 |
|---|---|
| `fig1_learning_curves.png/.pdf` | 学习曲线：五算法中位数 ± IQR；每个算法的逐 seed 曲线 |
| `fig2_final_by_seed.png/.pdf` | seed × 算法的最终成绩网格；每个 seed 达到阈值的步数 |
| `fig3_eval_states.png/.pdf` | 20 个固定评估起始状态上的逐状态回报（诊断图） |
| `tables.md` / `summary.csv` / `per_seed.csv` | 汇总表 / 聚合 / 逐 seed 明细 |
| `raw/{algo}_seed{n}.json` | 单次运行：逐回合训练回报、每个评估点的 20 个回合回报、最终评估、超参快照 |
| `models/` | actor 权重（后续鲁棒性 / 模仿学习可直接复用；**不入库**，由 `.gitignore` 排除，丢失后用 `run_all.py --force` 重新生成） |
| `logs/` | 每次运行的标准输出 |

## 目录说明

- `run_all.py`：编排；`run_one.py` / `run_one_onpolicy.py`：单次运行驱动；`analyze.py`：表格与图。
- `archive_v1/`：只剩一个说明。第一版（3 seeds、旧版 PPO/TRPO）的 PPO/TRPO 结论已作废，原因见 `经验总结.md` §4；
  旧文件可从 git 历史取回（命令见该目录的 README）。
- 算法代码在 `07_TRPO/Pendulum`、`08_PPO/Pendulum`；改动前的版本在各自的 `legacy_v1/`。

## 注意

- n=10 能看出分布形状，但仍不做显著性检验；差距在 1–2 分以内的算法（TD3 vs SAC vs DDPG 的中位数）不要排名。
- Pendulum 是教学环境，不是公认 benchmark；这套脚本的价值是跑通评估协议，为后续 MuJoCo / Meta-World 复用。
- 同一个 seed 在不同设备（CPU / CUDA）上不会复现同一条轨迹，见 `经验总结.md` §5。
