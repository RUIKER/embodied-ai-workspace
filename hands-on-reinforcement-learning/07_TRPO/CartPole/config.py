import torch

# 训练超参数
NUM_EPISODES = 500
HIDDEN_DIM = 128
GAMMA = 0.98
LMBDA = 0.95
CRITIC_LR = 1e-2
KL_CONSTRAINT = 0.0005
ALPHA = 0.5
SEED = 0

# 当前策略网络使用Categorical分布，因此环境必须是离散动作环境。
ENV_NAME = 'CartPole-v1'
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")