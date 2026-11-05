import numpy as np
import torch


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


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy().reshape(-1)
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.from_numpy(np.asarray(advantage_list, dtype=np.float32))


def compute_gae(gamma, lmbda, td_delta, ends):
    """多回合批数据的 GAE：在回合边界处切断优势的向后传播。

    td_delta: [T] 或 [T, 1]，delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)；
              时间上限造成的 truncated 仍然 bootstrap，所以 terminated 与 ends 是两回事。
    ends:     [T]，该步之后回合是否结束（terminated 或 truncated）。
    返回形状为 [T] 的优势序列。
    """
    td_delta = td_delta.detach().cpu().numpy().reshape(-1)
    ends = np.asarray(ends, dtype=np.float32).reshape(-1)
    advantage = np.zeros_like(td_delta)
    running = 0.0
    for t in reversed(range(len(td_delta))):
        running = td_delta[t] + gamma * lmbda * (1.0 - ends[t]) * running
        advantage[t] = running
    return torch.from_numpy(advantage.astype(np.float32))
