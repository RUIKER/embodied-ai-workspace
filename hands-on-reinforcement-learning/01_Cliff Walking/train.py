from config import (
    DISCOUNT_FACTOR,
    LEARNING_RATE,
    MAP_HEIGHT,
    MAP_LENGTH,
    NUM_ACTIONS,
    NUM_EPISODES,
    PRINT_INTERVAL,
)
from environment import Env
from q_learning import QTable as AlgorithmTable
# 如果要使用 SARSA，将上一行注释掉，并取消下一行的注释。
# from sarsa import SarsaTable as AlgorithmTable


def cliff_walk():
    """训练智能体通过悬崖地图，并输出每 20 回合的累计奖励。"""
    env = Env(length=MAP_LENGTH, height=MAP_HEIGHT)
    table = AlgorithmTable(
        length=MAP_LENGTH,
        height=MAP_HEIGHT,
        actions=NUM_ACTIONS,
        alpha=LEARNING_RATE,
        gamma=DISCOUNT_FACTOR,
    )
    for num_episode in range(NUM_EPISODES):
        # 每次循环代表一个完整的训练回合。
        episodic_reward = 0
        is_terminated = False
        s0 = [0, 0]
        while not is_terminated:
            # 在当前状态选择动作，与环境交互后更新 Q 值。
            action = table.take_action(s0[0], s0[1])
            r, s1, is_terminated = env.step(action)

            # 统一提前选择下一动作：SARSA 会使用它，Q-learning 会忽略它。
            next_action = None
            if not is_terminated:
                next_action = table.take_action(s1[0], s1[1])

            table.update(action, s0, s1, r, is_terminated, next_action)
            episodic_reward += r
            # env.render(frames=100)
            s0 = s1
        if num_episode % PRINT_INTERVAL == 0:
            print("Episode: {}, Score: {}".format(num_episode, episodic_reward))
        env.reset()