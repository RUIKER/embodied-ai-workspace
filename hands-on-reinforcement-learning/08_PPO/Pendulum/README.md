# PPO on Pendulum

在 `Pendulum-v1` 上使用连续动作 PPO。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

策略网络输出高斯分布的均值和标准差，动作采样后裁剪到环境边界。PPO 的 log probability 会沿动作维度求和，reward 会在 critic 更新时进行缩放。


**v2 变更（2026-10）**：每收集 `EPISODES_PER_UPDATE=10` 个完整回合更新一次，minibatch + 梯度裁剪；`dones` 只记 terminated，GAE 在回合边界切断，时间截断处 bootstrap；γ=0.99、λ=0.95。改动前的版本在 `legacy_v1/`，对比见 `../../experiments/seed_experiments/five_algorithms/经验总结.md` §4。
