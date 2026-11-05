import numpy as np
import gymnasium as gym


def make_environment(env_name):
    """创建车杆环境，并为DDPG提供一维连续动作代理空间。"""
    env = gym.make(env_name)
    if not hasattr(env.action_space, 'n') or env.action_space.n != 2:
        raise TypeError('当前车杆适配器要求两个离散动作。')
    state_dim = env.observation_space.shape[0]
    # actor 的连续输出范围设为[-1, 1]，之后映射为左/右离散动作。
    return env, state_dim, 1, np.array([-1.0]), np.array([1.0])


def continuous_to_discrete(action):
    """将DDPG的一维连续动作映射到CartPole的0/1动作。"""
    return int(np.asarray(action).reshape(-1)[0] >= 0.0)
