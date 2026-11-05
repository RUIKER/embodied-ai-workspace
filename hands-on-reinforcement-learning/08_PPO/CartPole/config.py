import torch

# actor 学习率：更新策略网络参数。
ACTOR_LR = 1e-3
# critic 学习率：拟合状态价值函数，通常可以和 actor 使用不同步长。
CRITIC_LR = 1e-2
NUM_EPISODES = 500
HIDDEN_DIM = 128
# gamma 控制未来奖励的折扣程度；lambda 控制 GAE 使用多步信息的程度。
GAMMA = 0.98
LMBDA = 0.95
# 同一条 episode 数据重复更新的次数，以及 PPO ratio 的截断范围。
EPOCHS = 10
EPS = 0.2
SEED = 0

ENV_NAME = 'CartPole-v1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
