import gymnasium as gym


def make_environment(env_name):
    """创建连续动作环境并返回动作边界。"""
    env = gym.make(env_name)
    if not hasattr(env.action_space, 'shape'):
        raise TypeError('SAC Pendulum需要Box连续动作空间。')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return (
        env,
        state_dim,
        action_dim,
        env.action_space.low.copy(),
        env.action_space.high.copy(),
    )
