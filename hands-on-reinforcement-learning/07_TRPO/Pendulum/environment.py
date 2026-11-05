import gymnasium as gym


def make_environment(env_name):
    """创建连续动作环境并返回状态维度、动作维度和动作边界。"""
    env = gym.make(env_name)
    if not hasattr(env.action_space, 'shape') or env.action_space.shape is None:
        raise TypeError('连续TRPO需要Box动作空间。')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return (
        env,
        state_dim,
        action_dim,
        env.action_space.low.copy(),
        env.action_space.high.copy(),
    )
