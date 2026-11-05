# REINFORCE

在 `CartPole-v1` 环境中使用 REINFORCE 学习离散动作策略。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

## 文件说明

- `config.py`：训练超参数和环境配置。
- `environment.py`：创建 Gymnasium 环境。
- `networks.py`：输出动作概率分布的策略网络。
- `agents.py`：REINFORCE 动作采样和策略梯度更新。
- `trainer.py`：按完整 episode 采样并更新策略。
- `visualization.py`：绘制回报曲线。
- `经验总结.md`：REINFORCE 原理、代码流程和排错记录。

REINFORCE 是 on-policy 算法，通常在一个完整 episode 结束后，根据折扣回报更新一次策略网络，不使用 DQN 风格的 replay buffer。
