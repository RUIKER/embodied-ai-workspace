# DDPG on Pendulum

标准 DDPG 连续动作 pipeline，使用 replay buffer、actor/critic target network 和软更新。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

## 手动切换探索噪声

编辑 `config.py`：

```python
NOISE_TYPE = 'normal'
```

或：

```python
NOISE_TYPE = 'ou'
```

- `normal`：每一步独立采样高斯噪声。
- `ou`：使用具有时间相关性的 Ornstein-Uhlenbeck 噪声。

OU 噪声参数：

```python
NOISE_SIGMA = 0.01
OU_THETA = 0.15
```

每个 episode 开始时会调用 `agent.reset_noise()`，清除 OU 噪声的历史状态。
