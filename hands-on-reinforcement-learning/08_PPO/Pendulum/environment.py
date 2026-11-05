import gymnasium as gym


def make_environment(env_name):
    env = gym.make(env_name)
    if not hasattr(env.action_space, 'shape') or env.action_space.shape is None:
        raise TypeError('连续PPO需要Box动作空间。')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return (
        env,
        state_dim,
        action_dim,
        env.action_space.low.copy(),
        env.action_space.high.copy(),
    )
