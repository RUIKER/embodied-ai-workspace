import torch

# actor/critic/alpha 分别有独立的优化器和学习率。
ACTOR_LR = 3e-4
CRITIC_LR = 3e-3
ALPHA_LR = 3e-4
NUM_EPISODES = 100
HIDDEN_DIM = 128
# gamma 折扣未来回报；tau 控制 target critic 的软更新速度。
GAMMA = 0.99
TAU = 0.005
# SAC 是off-policy，经验池可以保存很多历史transition。
BUFFER_SIZE = 100000
MINIMAL_SIZE = 1000
BATCH_SIZE = 64
REWARD_SHIFT = 8.0
REWARD_SCALE = 8.0
SEED = 0

ENV_NAME = 'Pendulum-v1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
