import matplotlib.pyplot as plt

from rl_utils import moving_average


def plot_training_results(return_list, env_name, window_size=9):
    """绘制原始回报和移动平均回报。

    TD3/DDPG 属于 off-policy 算法，训练回报受探索噪声影响较大，
    单条 episode 曲线会出现明显抖动。判断是否学会时应重点看移动平均。
    """
    episodes = list(range(len(return_list)))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, return_list, linewidth=1)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TD3 on {}'.format(env_name))

    plt.subplot(1, 2, 2)
    plt.plot(episodes, moving_average(return_list, window_size))
    plt.xlabel('Episodes')
    plt.ylabel('Moving average returns')
    plt.title('TD3 moving average')

    plt.tight_layout()
    plt.show()
