"""PyTorch Mountain Car training configuration."""

ENVIRONMENT_NAME = "MountainCarContinuous-v0"
NUM_EPISODES = 200
# 与 numpy 版一致使用 0.95：更短的等效 horizon 让 n 步 TD 的
# 价值传播更快，终点 +100 的信号能更快回传到轨迹前段。
DISCOUNT_FACTOR = 0.95
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
# 增大 entropy 系数以维持探索：MountainCarContinuous 在到达终点前
# 只有动作惩罚（-0.1*a²）信号，梯度会把动作和标准差一起压向 0，
# 太小的系数（如 1e-3）无法阻止探索塌缩。
ENTROPY_COEFFICIENT = 1e-2
# n 步回报的步数：把终点奖励更快地沿轨迹回传。
N_STEP = 5
# TD 更新内部的奖励缩放（+100 -> +1）：让 MLP critic 的目标值保持
# 在 O(1)，Adam 才能快速拟合。统计与曲线仍记录原始奖励。
REWARD_SCALE = 0.01
RANDOM_SEED = 42
SMOOTHING_WINDOW = 10
PRINT_INTERVAL = 10
