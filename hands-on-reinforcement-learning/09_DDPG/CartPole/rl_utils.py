import collections
import random

import numpy as np


class ReplayBuffer:
    """DDPG使用的长期off-policy经验回放池。"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(next_states), np.asarray(dones)

    def size(self):
        return len(self.buffer)


def moving_average(values, window_size):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window_size:
        return values
    cumulative_sum = np.cumsum(np.insert(values, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    radius = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(values[:window_size - 1])[::2] / radius
    end = (np.cumsum(values[:-window_size:-1])[::2] / radius)[::-1]
    return np.concatenate((begin, middle, end))
