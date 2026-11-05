# PPO 实验

本目录包含 PPO 在离散动作和连续动作环境中的两个独立实现。

| 目录 | 环境 | 策略分布 | 入口 |
| --- | --- | --- | --- |
| `CartPole` | `CartPole-v1` | `Categorical` | `CartPole/main.py` |
| `Pendulum` | `Pendulum-v1` | `Normal` | `Pendulum/main.py` |

两个 case 分别维护自己的配置、网络、agent、训练器和工具文件，避免离散动作与连续动作逻辑混在同一个脚本中。

## 运行车杆

```bash
cd CartPole
python -m pip install -r requirements.txt
python main.py
```

## 运行倒立摆

```bash
cd Pendulum
python -m pip install -r requirements.txt
python main.py
```

PPO 的公共部分是 clipped surrogate objective、GAE 和 critic 更新；两个 case 的主要区别是动作分布和 log probability 的计算方式。
