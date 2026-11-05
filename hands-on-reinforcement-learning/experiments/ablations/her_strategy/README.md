# HER 目标采样策略消融

对比 `future` / `final` / `episode` 三种 future-goal 选择策略，固定
`HER_RATIO=0.8`、10 seeds、2000 episodes、固定 20 个评估目标。

- `future`：复用 `experiments/seed_experiments/her_main` 的 10-seed 主实验。
- `final` / `episode`：本目录补跑。

## 运行

```bash
cd experiments/ablations/her_strategy
bash run_strategy_ablation.sh
../../../.venv/Scripts/python.exe aggregate_strategy.py
```

输出：`results/strategy_table.md`、`results/strategy_results.csv`、
`results/strategy_success_curves.png`、`results/strategy_final_success.png`。

## 结果（10 seeds）

| 策略 | 最终成功率均值 [95% bootstrap] | >=0.9 seeds | AUC | 收敛回合 |
| --- | ---: | ---: | ---: | ---: |
| future | 0.80 [0.50, 1.00] | 8/10 | 0.66 | 408（9/10） |
| final | 0.89 [0.70, 1.00] | 9/10 | 0.67 | 412（9/10） |
| episode | 1.00 [1.00, 1.00] | 10/10 | 0.85 | 326（10/10） |

实际结果和“future 最好”的预期相反。 `episode` 允许取过去/当前/下一步的目标，
包含更多“已经到达”的平凡目标，所以更容易；这不是泛化更好的证据。三者区间重叠，
n=10 不排序；`future` 仍是标准选择。详见 `../../../12_HER/经验总结.md` §3.4。

## 设计依据

- future 只取 t 之后的状态，目标一定可达且多样；final 只取终点，多样性差；episode 会取到过去的状态，可能产生已经达成的平凡目标。
- 评价时不只看最终成功率，还看曲线上升的时间、达标 seed 比例和未收敛 seed 的形态。三者最终都为 1.0 时，差异体现在 AUC 和收敛回合上。
- n=10 的区间仍然较宽，1–2 个百分点的差异不作强结论。
