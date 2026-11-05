# SAC on Pendulum

本目录将原始 SAC 示例拆分为完整 pipeline，只实现连续动作 `Pendulum-v1`。

## 运行

```bash
cd Pendulum
python -m pip install -r requirements.txt
python main.py
```

## 文件说明

- `config.py`：训练超参数。
- `environment.py`：创建 Pendulum 环境并读取动作边界。
- `networks.py`：高斯策略和 Q 网络。
- `agents.py`：双 Q、自动温度系数、soft Bellman target 和 SAC 更新。
- `trainer.py`：off-policy 交互和 replay buffer 更新。
- `rl_utils.py`：经验回放池和移动平均。
- `visualization.py`：回报曲线。
- `经验总结.md`：SAC 原理、代码流程和教程结果对比。
- `practice.py`：原始示例，仅作参考。
