# TRPO

本项目将原始 TRPO 示例拆分为配置、环境、网络、agent、训练器和可视化模块，在 `CartPole-v1` 上训练离散动作策略。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

## 文件说明

- `config.py`：TRPO 超参数和环境配置。
- `environment.py`：创建离散动作环境。
- `networks.py`：策略网络和价值网络。
- `agents.py`：GAE、共轭梯度、KL 约束线搜索和 TRPO 更新。
- `trainer.py`：按完整 episode 采样并更新 agent。
- `visualization.py`：绘制回报和移动平均曲线。
- `practice.py`：原始示例，保留作算法对照。
- `rl_utils.py`：移动平均和优势函数等工具。

当前策略使用 `Categorical` 分布，只支持离散动作环境。若要用于 `Pendulum-v1`，需要改用连续动作策略分布，例如高斯策略，并重新处理动作空间。
