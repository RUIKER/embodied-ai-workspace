import torch

NUM_EPISODES = 2000
HIDDEN_DIM = 128
GAMMA = 0.9
LMBDA = 0.9
CRITIC_LR = 1e-2
KL_CONSTRAINT = 0.00005
ALPHA = 0.5
REWARD_SHIFT = 8.0
REWARD_SCALE = 8.0
SEED = 0

ENV_NAME = 'Pendulum-v1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
