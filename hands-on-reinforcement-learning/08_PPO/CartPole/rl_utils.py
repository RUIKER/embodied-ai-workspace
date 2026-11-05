import numpy as np
import torch


def moving_average(values, window_size):
    """平滑回报曲线；训练 episode 不足窗口时直接返回原始值。"""
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window_size:
        return values
    cumulative_sum = np.cumsum(np.insert(values, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    radius = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(values[:window_size - 1])[::2] / radius
    end = (np.cumsum(values[:-window_size:-1])[::2] / radius)[::-1]
    return np.concatenate((begin, middle, end))


def compute_advantage(gamma, lmbda, td_delta):
    """从后往前计算 GAE 优势。

    td_delta[t] 是第 t 步的 TD 误差，递推式为：
    A[t] = td_delta[t] + gamma * lambda * A[t + 1]
    """
    td_delta = td_delta.detach().cpu().numpy().reshape(-1)
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.from_numpy(np.asarray(advantage_list, dtype=np.float32))
