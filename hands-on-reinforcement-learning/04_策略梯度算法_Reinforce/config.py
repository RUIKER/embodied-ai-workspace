import torch
# 学习率
LR = 1e-3
# 折扣因子
GAMMA = 0.98

NUM_EPISODES = 1000
HIDDEN_DIM = 128
ENV_NAME = 'CartPole-v1'
SEED = 0

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")