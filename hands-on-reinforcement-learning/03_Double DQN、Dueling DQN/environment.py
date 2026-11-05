import gymnasium as gym


def make_environment(env_name, action_dim):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    return env, state_dim, action_dim


def dis_to_con(discrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)