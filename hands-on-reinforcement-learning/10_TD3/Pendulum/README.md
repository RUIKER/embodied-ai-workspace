# TD3 on Pendulum

在连续动作环境 `Pendulum-v1` 上实现 **Twin Delayed DDPG (TD3)**。

TD3 是在 DDPG 基础上的三项改进：

1. **Clipped double-Q**：同时训练两个 critic，用二者的最小值计算 target，抑制 Q 值高估。
2. **Target policy smoothing**：给 target actor 的下一动作加截断噪声，平滑 critic 的 Q 函数。
3. **Delayed policy update**：critic 更新两次后，才更新一次 actor 和 target 网络。

## 运行

```bash
cd Pendulum
python -m pip install -r requirements.txt
python main.py
```

运行结束后会打印：

```text
最后 10 回合训练平均回报: ...
确定性策略评估回报: 均值 ..., 标准差 ..., 最佳 ...
```

并弹出原始回报曲线和移动平均曲线。

> 训练回报包含探索噪声，通常明显差于确定性评估回报。
> 判断算法是否学会时，应同时看移动平均曲线和无噪声评估结果。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 全部超参数，附参数含义与调参建议 |
| `environment.py` | 创建 `Pendulum-v1`，检查连续动作空间并读取动作边界 |
| `networks.py` | 确定性 Actor 和 Q Critic 网络 |
| `noise.py` | 高斯探索噪声 + target policy smoothing |
| `agents.py` | TD3 的 target 计算、双 critic 更新、延迟 actor 更新和软更新 |
| `trainer.py` | off-policy 训练循环与确定性策略评估 |
| `rl_utils.py` | replay buffer、移动平均 |
| `visualization.py` | 训练曲线 |
| `main.py` | 运行入口 |
| `practice.py` | 单文件教学版，便于对照 Spinning Up 阅读和调试 |
| `经验总结.md` | 原理、实现细节、实验结果与排错经验 |

## 推荐阅读顺序

1. `config.py`：先看清有哪些超参数。
2. `networks.py`：actor 和 critic 的输入输出是什么。
3. `agents.py` 的 `calc_target()`：TD3 的核心是 target 怎么算。
4. `agents.py` 的 `update()`：critic 与 actor 的更新顺序、延迟条件。
5. `trainer.py`：数据如何进入 buffer、何时触发更新。
6. `practice.py`：把整个算法压缩到一个文件后再通读一遍。

## 默认超参数

| 参数 | 值 | 说明 |
| --- | ---: | --- |
| `ACTOR_LR` | `1e-3` | actor 学习率 |
| `CRITIC_LR` | `1e-3` | 两个 critic 各自的学习率 |
| `NUM_EPISODES` | `200` | 训练回合数，每回合 200 步 |
| `HIDDEN_DIM` | `128` | 两层隐藏层宽度 |
| `GAMMA` | `0.99` | 折扣因子 |
| `TAU` | `0.005` | target 软更新系数 |
| `POLICY_DELAY` | `2` | 每 2 次 critic 更新，更新 1 次 actor |
| `TARGET_NOISE` | `0.2` | target smoothing 噪声（归一化动作单位） |
| `NOISE_CLIP` | `0.5` | target smoothing 噪声截断（归一化动作单位） |
| `EXPLORATION_SIGMA` | `0.1` | 行为策略探索噪声（环境动作单位） |
| `BUFFER_SIZE` | `100000` | replay buffer 容量 |
| `MINIMAL_SIZE` | `1000` | 开始学习前的最少样本数 |
| `BATCH_SIZE` | `100` | 每次更新的 batch 大小 |

其中 `TARGET_NOISE`、`NOISE_CLIP`、`POLICY_DELAY` 来自 Spinning Up 的默认值。

## 快速调试

第一次运行时，建议先改成：

```python
NUM_EPISODES = 1
MINIMAL_SIZE = 1
BATCH_SIZE = 8
```

确认以下链路正常后再恢复正式配置：

- `env.step()` 返回的是 5 元组（Gymnasium API）。
- buffer 能采样，`agent.update()` 能完成一次反向传播。
- 动作形状是 `[1]`，且在 `[-2, 2]` 内。
- 评估函数能跑完一个 episode。

## 常见误解

- **TD3 不能用于 CartPole 这类离散动作环境。** 它的 actor 输出连续动作，双 Q 和 target smoothing 也都定义在连续动作空间上。
- **探索噪声和 target smoothing 不是同一件事。** 前者加在行为动作上用于探索；后者加在 target 动作上用于平滑 critic，训练 actor 时不会使用。
- **`truncated` 不是终止。** Pendulum 每 200 步截断，写入 buffer 的 `done` 只能使用 `terminated`，否则会错误地停止 bootstrap。
