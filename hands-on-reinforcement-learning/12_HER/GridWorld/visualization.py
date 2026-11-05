"""训练结果可视化：回报曲线 + 固定评估集上的成功率曲线。

两张图的分工：
- 左图看"训练过程发生了什么"：细线是每回合原始回报，粗线是滑动平均。
  稀疏奖励下原始回报只有 −50 和 −(到达步数) 两档，单看细线像随机噪声。
- 右图看"策略到底会不会做任务"：固定目标集上的成功率，0~1 连续变化，
  是判断 HER 是否有效的**主指标**。
"""

import numpy as np
import matplotlib.pyplot as plt

from rl_utils import moving_average


def plot_training_results(return_list, success_history, env_name='GridWorld',
                          use_her=True, window_size=9, num_eval_goals=20,
                          save_path=None):
    """画训练结果。

    Args:
        return_list:     每回合训练回报，长度 = num_episodes。
        success_history: [(episode, success_rate), ...]，可能为空。
        env_name:        图标题用的环境名。
        use_her:         决定图例文字；也提醒读者这两张图来自哪一组实验。
        window_size:     回报滑动平均窗口。
        num_eval_goals:  成功率是在多少个固定目标上算的，写进标题。
        save_path:       给了就存图；None 时只显示（plt.show 会阻塞）。
    """
    episodes = np.arange(1, len(return_list) + 1)
    label = 'DDPG + HER' if use_her else 'DDPG (no HER)'

    # 1×2 布局：左回报、右成功率，共享横轴含义（Episode）。
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---------------------------------------------------------------- 左图
    # 原始回报：alpha 低一些，作为背景噪点。
    axes[0].plot(episodes, return_list, color='#1f77b4',
                 alpha=0.30, linewidth=1, label='raw return')
    # 滑动平均：趋势线。因果平均，不会"提前"画出未来才有的上升。
    axes[0].plot(episodes, moving_average(return_list, window_size),
                 color='#1f77b4', linewidth=2,
                 label='moving average (%d)' % window_size)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('%s on %s' % (label, env_name))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    # ---------------------------------------------------------------- 右图
    # success_history 只在 eval_interval > 0 时才有内容。
    if success_history:
        # zip(*pairs) 把 [(ep, s), ...] 拆成两个元组。
        xs, ys = zip(*success_history)
        axes[1].plot(xs, ys, color='#2ca02c', marker='o', markersize=3)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success rate')
    axes[1].set_title('Evaluation success rate (%d fixed goals)'
                      % num_eval_goals)
    # 成功率天然在 [0, 1]，留一点边距避免点和边框重合。
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    # 没有 save_path 时阻塞在当前窗口，直到手动关闭。
    plt.show()
