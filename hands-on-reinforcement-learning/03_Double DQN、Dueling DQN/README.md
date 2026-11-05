# DQN、Double DQN、Dueling DQN

本项目在 `Pendulum-v1` 环境中将连续动作离散为 11 个动作，统一实现并比较：

- `DQN`：目标网络选择并评估最大 Q 值。
- `DoubleDQN`：online network 选择动作，target network 评估动作。
- `DuelingDQN`：使用 `V(s)` 和 `A(s, a)` 两个分支构建 Q 值。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

## 切换算法

编辑 `config.py`：

```python
ALGORITHM = 'DQN'
```

可选值：

```text
DQN
DoubleDQN
DuelingDQN
```

## 文件说明

- `agents.py`：统一 agent、动作选择和 TD 更新。
- `networks.py`：普通 Q 网络和 Dueling Q 网络。
- `trainer.py`：训练循环和经验回放调用。
- `environment.py`：Pendulum 环境及离散动作到连续动作的映射。
- `visualization.py`：回报和最大 Q 值曲线。
- `经验总结.md`：算法原理、排错记录和实验建议。
