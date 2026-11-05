import torch

# =============================================================================
# TD3 超参数
# =============================================================================
# TD3 是在 DDPG 上的三项改进：
#   1. clipped double-Q：用两个 critic 的最小值计算 Bellman target，抑制 Q 值高估。
#   2. target policy smoothing：给 target actor 的下一动作加截断噪声，让 critic 更平滑。
#   3. delayed policy updates：critic 更新若干次后，才更新一次 actor 和 target 网络。
#
# 下面大多数默认值来自 Spinning Up 的 TD3 实现：
#   https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/td3

# actor 和 critic 使用相同学习率是 TD3 原始论文/Spinning Up 的默认设置。
# 注意：DDPG 教程常把 critic 学习率设得更大；TD3 的延迟更新已经缓解了
# actor 追着不准确 critic 跑的问题，因此不必刻意拉开两者学习率。
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3

# Pendulum 每个 episode 固定 200 步，200 个 episode 约 4 万次环境交互。
# 想快速验证代码链路时，先把 NUM_EPISODES 改成 1、MINIMAL_SIZE 改成 1。
NUM_EPISODES = 200

# 两层隐藏层，每层 128 个神经元。
HIDDEN_DIM = 128

# 折扣因子。Pendulum 的奖励范围约为 [-16, 0]，gamma 取 0.99 表示
# 智能体愿意为较远未来的回报付出努力。
GAMMA = 0.99

# target 网络软更新系数：
#   theta_target <- (1 - tau) * theta_target + tau * theta
# tau 越小，target 变化越慢，训练越稳定，但跟踪当前网络也越慢。
TAU = 0.005

# 延迟策略更新：每更新 POLICY_DELAY 次 critic，才更新一次 actor 和 target。
# 论文默认值为 2，即“critic 更新两次，actor 更新一次”。
POLICY_DELAY = 2

# target policy smoothing 使用的噪声。
# TARGET_NOISE 和 NOISE_CLIP 都是在“归一化动作空间” [-1, 1] 中定义的，
# agents.py 会再乘以动作半幅 action_scale，映射到 Pendulum 的 [-2, 2]。
TARGET_NOISE = 0.2
NOISE_CLIP = 0.5

# 行为策略的探索噪声标准差，单位是环境动作单位（Pendulum 中为 [-2, 2]）。
# 训练时执行 a = mu(s) + N(0, EXPLORATION_SIGMA^2)，再把动作裁剪到合法范围。
# 评估时关闭该噪声，执行纯确定性动作。
EXPLORATION_SIGMA = 0.1

# off-policy replay buffer 配置。
# Pendulum 的 transition 很小，10 万条只占很少内存。
BUFFER_SIZE = 100000
# buffer 中至少有这么多条数据后才开始采样，避免早期用极小样本过拟合。
MINIMAL_SIZE = 1000
BATCH_SIZE = 100

# 训练结束后用确定性策略额外评估多少个 episode。
# 评估 episode 太少时方差很大，20 个是精度与耗时的折中。
EVAL_EPISODES = 20
# 评估专用种子，与训练种子解耦：比较不同训练种子时，
# 所有模型都在同一批初始状态上评估。
EVAL_SEED = 10000

SEED = 0
ENV_NAME = 'Pendulum-v1'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
