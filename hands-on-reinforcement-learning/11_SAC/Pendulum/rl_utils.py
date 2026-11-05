import collections
import random

import numpy as np


class ReplayBuffer:
    """SAC使用的off-policy经验回放池。"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # 每条经验保存(s, a, r, s', terminal)。
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机采样打散相邻时间步的相关性，提高off-policy训练稳定性。
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.asarray(states), np.asarray(actions), np.asarray(rewards),
            np.asarray(next_states), np.asarray(dones),
        )

    def size(self):
        return len(self.buffer)


def moving_average(values, window_size):
    """平滑训练回报；样本不足一个窗口时返回原始序列。"""
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window_size:
        return values
    cumulative_sum = np.cumsum(np.insert(values, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    radius = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(values[:window_size - 1])[::2] / radius
    end = (np.cumsum(values[:-window_size:-1])[::2] / radius)[::-1]
    return np.concatenate((begin, middle, end))
