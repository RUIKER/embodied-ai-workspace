# DDPG 实验

本目录包含 DDPG 在两个环境中的 pipeline：

| 目录 | 环境 | 说明 |
| --- | --- | --- |
| `Pendulum` | `Pendulum-v1` | 标准连续动作 DDPG，主要实验 case |
| `CartPole` | `CartPole-v1` | 连续 actor 映射到左右离散动作的教学适配 |

两个 case 都支持在 `config.py` 中手动切换探索噪声：

```python
NOISE_TYPE = 'normal'
```

或：

```python
NOISE_TYPE = 'ou'
```

## 运行

```bash
cd Pendulum
python -m pip install -r requirements.txt
python main.py
```

车杆版本：

```bash
cd CartPole
python -m pip install -r requirements.txt
python main.py
```

DDPG 是 off-policy 算法，使用 replay buffer；actor 是确定性策略，探索依赖动作噪声；critic 和 actor 都有 target network，并使用软更新。
