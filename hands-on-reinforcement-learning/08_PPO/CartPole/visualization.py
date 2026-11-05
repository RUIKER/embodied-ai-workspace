import matplotlib.pyplot as plt

from rl_utils import moving_average


def plot_training_results(return_list, env_name):
    episodes = list(range(len(return_list)))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.subplot(1, 2, 2)
    plt.plot(episodes, moving_average(return_list, 9))
    plt.xlabel('Episodes')
    plt.ylabel('Moving average returns')
    plt.title('PPO moving average')
    plt.tight_layout()
    plt.show()
