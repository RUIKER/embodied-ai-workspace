# Mountain Car Actor-Critic with PyTorch

这是原始 Mountain Car 项目的 PyTorch 版本。原版本使用 NumPy 手动更新线性 Actor-Critic；本版本使用 PyTorch 的 MLP、自动求导和 Adam 优化器。

## 运行

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

训练结果会保存到 `output/`：

- `episode_stats.csv`: 每个 episode 的奖励和步数
- `training_curves.png`: 训练曲线
- `actor_critic.pt`: Actor 和 Critic 的模型参数

## 主要实现

- `networks.py`: Actor 和 Critic 神经网络
- `trainer.py`: episode 批量更新的 n 步 TD Actor-Critic
- `environment.py`: Gymnasium 环境适配
- `main.py`: 实验入口和模型保存

## 关键实现细节

1. **terminated 与 truncated 分开处理**：只有真正到达终点（terminated）才停止
   bootstrap；999 步时间限制截断（truncated）的状态仍有价值，继续 bootstrap。
   把截断当终止会在每个未成功的 episode 结尾向 critic 注入错误的"终态"信号。
2. **动作用 clip 而不是 tanh**：对高斯采样直接 clip 到 [-1, 1]（与 numpy 版一致）。
   clip 会把相当一部分动作饱和到 ±1，形成 bang-bang 式摆动，小车才有机会积累
   动能爬上山坡发现 +100 奖励；tanh 压缩的零均值高斯探索几乎永远到不了终点，
   策略会收敛到"谷底不动、只省力气"的局部最优（reward 曲线缓慢升到 -5 左右
   但 episode 永远是 999 步——这是本项目早期版本的典型假学习曲线）。
3. **episode 结束后批量更新 + n 步回报**：逐步（per-step）更新时，MLP critic 用
   Adam 学习 +100 的终点价值极慢，且 Adam 的二阶矩归一化把小 advantage 步骤的
   噪声梯度放大到与关键步骤同量级，策略行为完全被探索噪声主导（标准差还会被
   "成功的都是极端动作"棘轮式顶到上限）。批量更新降低梯度方差，n 步回报把
   终点信号一次回传 n 步，信用传播显著加快。
4. **reward scale（仅用于 TD 更新）**：把终点 +100 缩放到 +1，critic 目标值保持
   O(1)，MLP+Adam 能快速拟合；`episode_stats.csv` 和曲线仍记录原始奖励，
   单位与 numpy 版直接可比。
5. **探索参数**：初始 std=1.0（`log_std=0`），entropy 系数 1e-2，γ=0.95（与
   numpy 版一致），防止动作惩罚梯度（到达终点前唯一信号是 -0.1·a²）把探索压垮。
