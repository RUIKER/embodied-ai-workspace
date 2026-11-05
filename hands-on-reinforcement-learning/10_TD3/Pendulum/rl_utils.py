import collections
import random

import numpy as np


class ReplayBuffer:
    """off-policy 经验回放池。

    每条数据是 (s, a, r, s', done)：
        done 在写入时只使用 terminated，不使用 truncated。
    deque(maxlen=capacity) 会在容量满时自动丢弃最旧的数据。
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机均匀采样，打破连续 transition 之间的时间相关性。
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        # 统一转成 NumPy 数组，方便 agents.py 一次性转成 batch tensor。
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)


def moving_average(values, window_size):
    """计算移动平均，用于平滑训练回报曲线。

    两端不足一个窗口时使用逐步增大的窗口，避免曲线开头缺失。
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window_size:
        return values
    cumulative_sum = np.cumsum(np.insert(values, 0, 0))
    middle = (
        cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    ) / window_size
    radius = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(values[:window_size - 1])[::2] / radius
    end = (np.cumsum(values[:-window_size:-1])[::2] / radius)[::-1]
    return np.concatenate((begin, middle, end))
