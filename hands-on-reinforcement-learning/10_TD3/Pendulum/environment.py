import gymnasium as gym


def make_environment(env_name):
    """创建连续动作环境，并读取状态/动作维度与动作上下界。

    TD3 只适用于连续动作空间，因此这里显式检查 action_space 是否为 Box。
    如果误传入 CartPole 之类的离散环境，应当尽早报错，而不是让
    actor 输出一个无法执行的浮点动作。
    """
    env = gym.make(env_name)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(
            'TD3 需要连续动作空间（gym.spaces.Box），'
            '当前环境是 {}。'.format(type(env.action_space).__name__)
        )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # copy() 防止后续对数组的原地修改影响环境自身的 action_space。
    action_low = env.action_space.low.copy()
    action_high = env.action_space.high.copy()
    return env, state_dim, action_dim, action_low, action_high
