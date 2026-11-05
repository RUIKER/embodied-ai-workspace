import matplotlib.pyplot as plt

from rl_utils import moving_average


def plot_training_results(return_list, env_name, window_size=9):
    episodes = list(range(len(return_list)))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.subplot(1, 2, 2)
    plt.plot(episodes, moving_average(return_list, window_size))
    plt.xlabel('Episodes')
    plt.ylabel('Moving average returns')
    plt.title('SAC moving average')
    plt.tight_layout()
    plt.show()
