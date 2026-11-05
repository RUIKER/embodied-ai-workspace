# her_main

DDPG vs DDPG+HER 的主对比（10 seeds，GridWorld）。

| 项 | 设置 |
| --- | --- |
| 环境 | 二维 GridWorld，目标 `[3.5,4.5]^2`，每回合 50 步 |
| 训练 | 2000 回合；buffer ≥ 200 条轨迹；每条轨迹 20 次更新；batch 256 |
| 评估 | 固定 20 个目标，`EVAL_SEED=10000`，确定性 actor |
| seeds | 0–9 |
| 设置 | HER ratio=0.8 / future；无 HER（ratio=0，纯 DDPG） |

## 运行

```bash
cd experiments/seed_experiments/her_main
bash run_all.sh
../../../.venv/Scripts/python.exe aggregate.py
```

## 结果

- `results/raw/*.json`：每个 seed 的逐回合回报、成功率历史、最终评估、耗时
- `results/success_curves.png`：成功率曲线（中位数 + 单 seed）
- `results/runs.csv` / `results/summary_table.md`：明细与聚合
- 结论、失败模式、debug 记录：`../../../12_HER/经验总结.md`
