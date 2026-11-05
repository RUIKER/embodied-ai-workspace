# DDPG on CartPole

这是一个教学适配版本：CartPole 原生是离散动作环境，而标准 DDPG 面向连续动作。因此 actor 输出一个 `[-1, 1]` 的连续信号，再映射为：

```text
连续输出 < 0 -> action 0
连续输出 >= 0 -> action 1
```

这可以帮助理解 DDPG 的 actor、critic、target network、replay buffer 和探索噪声，但不代表标准连续动作 DDPG benchmark。正式 DDPG 实验应参考 `../Pendulum`。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

在 `config.py` 中手动切换：

```python
NOISE_TYPE = 'normal'
# 或 NOISE_TYPE = 'ou'
```
