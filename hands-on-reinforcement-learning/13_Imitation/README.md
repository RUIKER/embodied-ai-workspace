# 13_Imitation：模仿学习

本目录记录三种模仿学习方法的**算法卡片**和 BC 的**最小实现**。

## 文件

| 文件 | 内容 |
|---|---|
| `算法卡片_BC.md` | Behavior Cloning：MLE=MSE、compounding error、多模态平均 |
| `算法卡片_ACT.md` | Action Chunking Transformer：action chunking + CVAE，推理 latent 置 0 |
| `算法卡片_DP.md` | Diffusion Policy：条件扩散、DDIM 加速、与 ACT 的对比 |
| `bc_minimal.py` | 最小 BC 实现：规则专家 + 两层 MLP，两个任务测两种失败 |

## 结果

见 `../results/`：

- `bc_grid.csv`：GridWorld 任务，不同示范条数下的训练 MSE 和闭环成功率。
- `bc_corridor.csv`：Corridor 任务，验证"多模态动作被单峰高斯平均"现象。
- `bc_curve.png`：两张任务的成功率对比曲线。

## 一句话结论

BC 在**误差可自我纠正**的任务上数据效率高，在**多模态动作**任务上会被 MSE 平均成错误动作；
ACT 和 Diffusion Policy 分别用 action chunking 和扩散模型解决这两个问题。
