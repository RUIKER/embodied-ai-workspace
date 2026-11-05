import torch

# Pendulum 每回合 200 步；2000 回合 = 400k env steps。
NUM_EPISODES = 2000
# 每收集这么多完整回合才更新一次（2000 步一批）。单回合 200 步的批太小、
# 样本高度相关，是旧版 PPO 在 Pendulum 上学得很差的主要原因。
EPISODES_PER_UPDATE = 10
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
HIDDEN_DIM = 128
GAMMA = 0.99
LMBDA = 0.95
EPOCHS = 10
MINIBATCH_SIZE = 64
# ratio 被限制在 [1-EPS, 1+EPS]，这是 PPO-Clip 的核心超参数。
EPS = 0.2
MAX_GRAD_NORM = 0.5
# 只缩放 critic 的训练目标，不改变日志中的原始环境回报。
REWARD_SHIFT = 8.0
REWARD_SCALE = 8.0
SEED = 0

ENV_NAME = 'Pendulum-v1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
