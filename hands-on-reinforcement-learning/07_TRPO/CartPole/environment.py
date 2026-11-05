

import gymnasium as gym


def make_environment(env_name):
    """创建离散动作环境，并返回状态维度和动作数量。"""
    env = gym.make(env_name)
    if not hasattr(env.action_space, 'n'):
        raise TypeError(
            'TRPO当前使用Categorical策略，只支持离散动作空间。'
        )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return env, state_dim, action_dim