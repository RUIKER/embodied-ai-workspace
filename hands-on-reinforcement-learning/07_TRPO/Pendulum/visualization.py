import matplotlib.pyplot as plt

from rl_utils import moving_average


def plot_training_results(return_list, env_name):
    episodes = list(range(len(return_list)))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('TRPO on {}'.format(env_name))
    plt.subplot(1, 2, 2)
    values = moving_average(return_list, 9)
    plt.plot(episodes, values)
    plt.xlabel('Episodes')
    plt.ylabel('Moving average returns')
    plt.title('TRPO moving average')
    plt.tight_layout()
    plt.show()
