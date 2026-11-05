"""全部超参集中在这里；实验脚本只读不改逻辑。

读代码前先看这个文件的三个常量，就能知道一次实验大概跑什么：

    USE_HER   / HER_RATIO / GOAL_STRATEGY     ← 消融就改这三个
    NUM_EPISODES / MINIMAL_EPISODES / N_TRAIN ← 决定训练量与耗时
    EVAL_GOALS / EVAL_INTERVAL                ← 决定成功率曲线的横轴密度
"""

import torch

# =============================================================================
# HER 超参
# =============================================================================
# HER（Hindsight Experience Replay）不是新算法，是 replay buffer 的改造：
# 采样时以 HER_RATIO 的概率把目标 g 换成同一条轨迹里 t 之后真实到达过的位置，
# 再用新目标重算奖励和 done。失败轨迹因此变成"以失败点为目标的成功轨迹"。
#
# 默认值来自《动手学强化学习》第 19 章，改之前先复现一遍。

# 两个网络分开的学习率，小任务上 1e-3 是 DDPG 的常用起点。
# 若 critic loss 剧烈震荡，先降到 3e-4；actor 学得慢可以单独调大。
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
# 隐藏层宽度。128 足够表达"位置 + 目标 → 动作"这种低维映射。
HIDDEN_DIM = 128

# 状态是 [x, y, goal_x, goal_y]，动作是二维平面位移。
# 注意 STATE_DIM = 4 已经包含目标；网络不知道哪两维是目标（见 networks.py）。
STATE_DIM = 4
ACTION_DIM = 2
# 动作每一维的上下限，同时用于 Actor 输出缩放和 buffer 前的 clip。
ACTION_BOUND = 1.0

# 折扣因子。0.98 的有效视野约 1/(1−γ) = 50 步，正好等于 MAX_STEPS：
# 这意味着最后一步的回报不会被过度折扣掉，对"50 步内到达"的任务是合适的。
GAMMA = 0.98
# target 网络软更新系数。1/τ = 200，target 大约每 200 次更新跟随一次 online。
TAU = 0.005
# 行为策略的探索噪声（作用在归一化动作上，范围 [-1, 1]）
# 0.1 表示每维平均 10% 的扰动；太大变成随机游走，太小则同一目标反复走同一条路。
SIGMA = 0.1

# ------------------------------------------------------------------ 训练量
NUM_EPISODES = 2000
# buffer 里攒够这么多条**轨迹**才开始更新。
# off-policy 的过渡在 Pendulum 里是 1000 条 transition，
# 这里单位是整条轨迹，差别很大：200 条轨迹 ≈ 最多 10000 条 transition。
# 之所以要 warmup，是因为 HER 需要 buffer 里有足够多不同轨迹，
# 否则假目标的分布过于单一。
MINIMAL_EPISODES = 200
# 每采一条轨迹，做多少次梯度更新。
# 总更新次数 ≈ N_TRAIN × (NUM_EPISODES − MINIMAL_EPISODES) = 20 × 1800 = 36000。
N_TRAIN = 20
BATCH_SIZE = 256
# buffer 存的是轨迹对象，一条轨迹最多 MAX_STEPS 个 transition
# 所以 10000 条轨迹最多约 50 万个 transition，内存占用很小。
BUFFER_SIZE = 10000

# ------------------------------------------------------------------ 环境
# 单回合最大步数；达到即 truncated（不是 terminated，HER 不会重算它）。
MAX_STEPS = 50
# 到达半径。5×5 平面上 0.15 很小，是"稀疏奖励"的直接来源。
DISTANCE_THRESHOLD = 0.15

# ------------------------------------------------------------------ HER 开关
# 消融实验就靠这两个开关，别改代码结构。
USE_HER = True
# 0.0 = 纯 DDPG（对照组）；1.0 = 全部用假目标
# 0.8 是原论文的推荐值：保留 20% 真实目标样本，避免策略完全偏离真实任务分布。
HER_RATIO = 0.8

# 目标采样策略：'future' / 'final' / 'episode'
# future：从 t 之后的时刻取（教程默认，效果最好）；目标一定可达且非平凡
# final ：只取轨迹最后一个状态，多样性差
# episode：整条轨迹任意时刻，会取到"已经达成"的平凡目标
GOAL_STRATEGY = 'future'

# ------------------------------------------------------------------ 评估
# 固定评估集：同一批 (起点=原点, 目标) 组合，所有算法 / 所有 seed / 所有评估点共用。
# 指标以**成功率**为主——稀疏奖励下回报几乎全是 -50（全程失败）或 -5 左右
# （到达即结束），没有中间地带，看回报曲线很难判断策略好坏。
EVAL_SEED = 10000
EVAL_GOALS = 20
EVAL_INTERVAL = 100   # 每多少个训练回合评估一次

SEED = 0
# 有 CUDA 就用 GPU；这个任务的网络很小，CPU 也就慢 2~3 倍。
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
