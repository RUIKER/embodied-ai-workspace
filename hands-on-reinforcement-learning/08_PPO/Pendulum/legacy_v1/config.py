import torch

# 连续动作环境使用更小的 actor 学习率，避免策略变化过快。
ACTOR_LR = 1e-4
CRITIC_LR = 5e-3
NUM_EPISODES = 2000
HIDDEN_DIM = 128
GAMMA = 0.9
LMBDA = 0.9
EPOCHS = 10
# ratio 被限制在 [1-EPS, 1+EPS]，这是 PPO-Clip 的核心超参数。
EPS = 0.2
# 只缩放 critic 的训练目标，不改变日志中的原始环境回报。
REWARD_SHIFT = 8.0
REWARD_SCALE = 8.0
SEED = 0

ENV_NAME = 'Pendulum-v1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
