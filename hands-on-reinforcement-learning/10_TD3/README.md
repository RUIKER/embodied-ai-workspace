# TD3 实验

本目录实现 **Twin Delayed DDPG (TD3)**。

TD3 只适用于**连续动作空间**，因此这里只提供 `Pendulum-v1` 案例，不提供 CartPole 适配版。把 TD3 强行套到离散动作上需要额外的动作量化或分布改造，容易把算法细节和适配器混在一起，不利于教学。

## 目录结构

```text
10_TD3/
├── README.md
├── 经验总结.md
└── Pendulum/
    ├── config.py
    ├── environment.py
    ├── networks.py
    ├── noise.py
    ├── agents.py
    ├── trainer.py
    ├── rl_utils.py
    ├── visualization.py
    ├── main.py
    ├── practice.py
    ├── requirements.txt
    ├── README.md
    └── 经验总结.md
```

## 运行

```bash
cd Pendulum
python -m pip install -r requirements.txt
python main.py
```

## TD3 与 DDPG 的区别

| 组件 | DDPG | TD3 |
| --- | --- | --- |
| Critic 数量 | 1 个 online + 1 个 target | 2 个 online + 2 个 target |
| Bellman target | `Q_target(s', mu_target(s'))` | `min(Q1_target, Q2_target)`，动作先做 smoothing |
| Actor 更新频率 | 每个 critic step | 每 `policy_delay` 个 critic step |
| Target 更新频率 | 每个 critic step | 与 actor 同步 |
| 探索噪声 | Normal 或 OU | 默认独立高斯（OU 可选但不必要） |

三项改进分别针对：

1. Q 值高估 → 双 Q 取最小。
2. deterministic target 动作附近的 Q 尖峰 → target policy smoothing。
3. actor 追着不准确 critic 更新 → 延迟策略更新。

## 参考

- 原始论文：[Fujimoto, van Hoof & Meger, *Addressing Function Approximation Error in Actor-Critic Methods*, ICML 2018](https://arxiv.org/abs/1802.09477)
- 教程：[Twin Delayed DDPG — Spinning Up](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- 参考实现：[openai/spinningup td3.py](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/td3)
- 代码组织与课程背景：[《动手学强化学习》](https://hrl.boyuai.com/)

本实现保留了 Spinning Up 的算法结构，但改成了与仓库其他算法一致的拆分方式，并适配 Gymnasium 的 `reset()` / `step()` API。
