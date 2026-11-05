import torch

ACTOR_LR = 3e-4
CRITIC_LR = 3e-3
NUM_EPISODES = 200
HIDDEN_DIM = 64
GAMMA = 0.98
TAU = 0.005
BUFFER_SIZE = 10000
MINIMAL_SIZE = 1000
BATCH_SIZE = 64
NOISE_TYPE = 'normal'  # 可选：'normal' 或 'ou'
NOISE_SIGMA = 0.01
OU_THETA = 0.15
SEED = 0

ENV_NAME = 'Pendulum-v1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
