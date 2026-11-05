import torch


# 手动选择算法：DQN、DoubleDQN 或 DuelingDQN。
# DQN：目标网络同时负责选择最大动作和计算其Q值。
# DoubleDQN：在线网络选择动作，目标网络计算该动作的Q值。
# DuelingDQN：网络内部将Q值拆成V(s)和A(s,a)两个分支。
ALGORITHM = 'DuelingDQN'

# 学习率：每次梯度更新时，online network参数移动的步长。
LR = 1e-2
# 折扣因子：越接近1，越重视未来奖励。
GAMMA = 0.98
# 探索概率：以该概率随机选动作，否则选择当前最大Q值动作。
EPSILON = 0.01
# 每隔多少次更新，把online network复制给target network。
TARGET_UPDATE = 50

NUM_EPISODES = 200
HIDDEN_DIM = 128
BUFFER_SIZE = 5000
MINIMAL_SIZE = 1000
BATCH_SIZE = 64
SEED = 0

ENV_NAME = 'Pendulum-v1'
ACTION_DIM = 11
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")