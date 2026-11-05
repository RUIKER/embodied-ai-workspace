"""HalfCheetah SAC 的超参与训练设置。

超参对齐常见的 MuJoCo SAC 配置（SB3 / SAC 论文量级），方便和公开参考值对照。
每一项都写了“为什么是这个值”，改之前先想清楚会影响哪条曲线。
"""

import torch

ENV_ID = 'HalfCheetah-v5'
# 只跑 3 个 seed：1M steps 单进程约 3.9h，3 个并行已经是这台机器的上限；
# 3 个点足以看出 seed 间方差，不足以做统计检验。
SEEDS = [0, 1, 2]

# 1M 是 MuJoCo 连续控制 benchmark 的通用预算（SAC/TD3 论文、SB3 zoo 都用这个量级），
# 用它才能和公开数字对比；更短的训练只能比较样本效率。
TOTAL_STEPS = 1_000_000
# 2×256 是 MuJoCo SAC 的常用容量：HalfCheetah 的 obs 只有 17 维，
# 更宽的网络收益很小，更窄（如单层 64）在 1M 步内学不满。
HIDDEN_DIM = 256
# actor / critic / alpha 用同一个学习率，是 SAC 原论文和 SB3 的默认做法。
# Pendulum 那种小环境用 1e-3 也能学，MuJoCo 上 3e-4 更稳。
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
# gamma=0.99 对应约 100 步的有效视野（控制步长 0.05s，约 5 秒）；
# HalfCheetah 要跑满 1000 步，视野太短会让它只顾眼前速度。
GAMMA = 0.99
# Polyak 系数：target 网络每步只向 online 网络靠近 0.5%，避免 Q 目标自身漂移。
TAU = 0.005
# batch 256 是 SAC 的默认值；太小梯度噪声大，太大单步成本高且探索不足。
BATCH_SIZE = 256
# buffer 能装下整段 1M 训练，等价于“不淘汰旧数据”；
# 显存/内存占用约 17+6+1+17+1 个 float × 1e6 ≈ 170MB。
BUFFER_SIZE = 1_000_000
# 前 10k 步只采数据不更新：buffer 太小的时候 batch 相关性极强，先更新反而拖慢。
MINIMAL_SIZE = 10_000

# HalfCheetah 的 reward 本身在 0–10 量级，不需要 Pendulum 那种 shift/scale。
# 注意：reward_scale 在 SAC 里和 alpha 是耦合的（缩放 reward 等价于缩放温度），
# 这里保持 1.0 就是为了让 alpha 的含义和公开实现一致。
REWARD_SHIFT = 0.0
REWARD_SCALE = 1.0

# 每 20k 步评估一次：1M 共 50 个点，足够画出曲线形状，又不至于拖慢训练。
EVAL_INTERVAL = 20_000
EVAL_EPISODES = 5
FINAL_EVAL_EPISODES = 20
# 评估用的随机种子和训练 seed 分开，保证所有 seed 面对同一串初始状态。
EVAL_SEED = 10_000

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
