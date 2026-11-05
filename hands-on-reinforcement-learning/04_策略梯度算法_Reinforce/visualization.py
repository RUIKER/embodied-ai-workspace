import matplotlib.pyplot as plt
import rl_utils

def plot_returns(return_list):
    episodes_list = list(range(len(return_list)))
    if len(return_list) >= 9:
        values = rl_utils.moving_average(return_list, 9)
    else:
        values = return_list
    plt.plot(episodes_list, values)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on CartPole-v1')
    plt.show()