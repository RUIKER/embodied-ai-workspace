import torch

NUM_EPISODES = 2000
# 每收集这么多完整回合才更新一次；单回合 200 步的批太小，梯度和 Fisher 估计都很噪。
EPISODES_PER_UPDATE = 10
HIDDEN_DIM = 128
GAMMA = 0.99
LMBDA = 0.95
CRITIC_LR = 1e-3
CRITIC_EPOCHS = 10
MINIBATCH_SIZE = 64
# 旧值 5e-5 比常用的 0.01 小约 200 倍，策略几乎不动。
KL_CONSTRAINT = 0.01
ALPHA = 0.5
REWARD_SHIFT = 8.0
REWARD_SCALE = 8.0
SEED = 0

ENV_NAME = 'Pendulum-v1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
