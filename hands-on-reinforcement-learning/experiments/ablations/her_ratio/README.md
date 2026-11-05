# experiments / ablations / her_ratio

HER（Hindsight Experience Replay）的消融实验目录。

- 算法实现：`12_HER/GridWorld/`
- 主对比（DDPG vs DDPG+HER，10 seeds）：结果在 `../../seed_experiments/her_main/results/`
- 本目录：**her_ratio 消融**（周四任务），以及后续的采样策略 / SAC+HER 消融

## 协议

| 项 | 设置 |
| --- | --- |
| 环境 | 二维 GridWorld，目标 `[3.5,4.5]^2`，每回合 50 步 |
| 评估 | 固定 20 个目标，`EVAL_SEED=10000`，确定性 actor |
| 指标 | 成功率（主）、峰值、AUC、收敛回合、最终成功率 |
| 训练 | 2000 回合；buffer ≥ 200 条轨迹；每条轨迹 20 次更新；batch 256 |
| 评估频率 | 每 50 回合（比计划的 100 更密，曲线更平滑） |
| seeds | 0–9（10 个）；主对比同 seed 复用 |
| 并行 | 10 个子进程；`OMP_NUM_THREADS=1` |

## her_ratio 消融

`ratio=0` 复用主对比的 `noher_s*.json`；`ratio=0.8` 复用
`her_r0.8_future_s*.json`；只新跑 `0.2 / 0.5 / 1.0`。

```bash
cd experiments/ablations/her_ratio
bash run_ratio_ablation.sh
../../../.venv/Scripts/python.exe aggregate_ratio.py
```

输出：

| 文件 | 内容 |
| --- | --- |
| `results/raw/ratio{r}_s{seed}.json` | 新跑的 ratio 0.2/0.5/1.0 |
| `results/ratio_results.csv` | 每个 ratio 的逐 seed 结果 |
| `results/ratio_table.md` | 中位数 / IQR / ≥0.9 seeds |
| `results/ratio_success_curves.png` | 成功率 vs 回合，中位数 + IQR 带 |
| `results/ratio_final_success.png` | ratio vs 最终成功率（箱线 + 单 seed） |

## 结果（10 seeds）

| ratio | 最终成功率均值 [95% bootstrap] | 最终>=0.9 的 seed | 收敛回合（回报，未收敛计 >2000） |
| ---: | ---: | ---: | ---: |
| 0（无 HER） | 0.66 [0.40, 0.92] | 6/10 | 1120 [657, >2000]（6/10 收敛） |
| 0.2 | 0.90 [0.70, 1.00] | 9/10 | 367 [344, 1105]（10/10 收敛） |
| 0.5 | 0.82 [0.56, 1.00] | 8/10 | 438 [320, 663]（8/10 收敛） |
| 0.8 | 0.80 [0.50, 1.00] | 8/10 | 410 [317, 522]（9/10 收敛） |
| 1.0 | 0.13 [0.06, 0.19] | 0/10 | >2000（0/10 收敛） |

可以下的结论：有 HER（0.2–0.8）都比无 HER 好；ratio=1.0 一定失败。
**不能下的结论**：0.2 / 0.5 / 0.8 之间谁更好——置信区间高度重叠。

## 方法说明（相对第一版的修正）

- **5 个 seed 改成 10 个**：第一版 seeds 0–4 上 ratio=0.8 是 5/5，扩到 10 个后是 8/10；
  同一设置在 seeds 0–4 与 5–9 上均值分别是 1.00 和 0.60。n=5 的结论不可靠。
- **聚合不再用中位数 / IQR**：0/1 型数据会退化成阶梯，且 n=10 的 IQR 没有统计意义；
  改为 bootstrap 95% 区间和 Wilson 区间。
- **收敛回合不再排除未收敛的 run**：未收敛按 `>2000` 计入分位数，并标注 k/n。
- **增加峰值 / AUC 指标**：避免“到过 1.0 又崩”被终点快照记成 0。

## 诊断：ratio=1.0 为什么失败

`diagnose_ratio1.py` 在 ratio=1.0 / 0.8、seeds 0–1 上记录训练目标分布、探索覆盖和宽网格成功率。
结论见 `../../../12_HER/经验总结.md` §3.3。

```bash
../../../.venv/Scripts/python.exe diagnose_ratio1.py --ratio 1.0 --seed 0
../../../.venv/Scripts/python.exe diagnose_ratio1.py --ratio 0.8 --seed 0
```

输出：`results/diagnostics/ratio{r}_s{seed}.json`。
