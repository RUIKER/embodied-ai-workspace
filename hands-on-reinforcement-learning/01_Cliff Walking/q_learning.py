import random

from config import EXPLORATION_RATE

class QTable:
    """Q-learning 的 Q 表和动作选择逻辑。"""

    def __init__(self, length, height, actions=4, alpha=0.1, gamma=0.9):
        # 将三维的 Q(s, a) 表展平成一维列表，初始值全部设为 0。
        self.table = [0] * actions * length * height
        self.actions = actions
        self.length = length
        self.height = height
        self.alpha = alpha
        self.gamma = gamma

    def _index(self, a, x, y):
        """Return the index of Q([x,y], a) in Q_table."""
        return a * self.height * self.length + x * self.length + y

    def _epsilon(self):
        """返回 epsilon-greedy 策略中的随机探索概率。"""
        return EXPLORATION_RATE

    def take_action(self, x, y):
        """使用 epsilon-greedy 策略选择动作。"""
        if random.random() < self._epsilon():
            # 随机探索：从所有合法动作中随机选择一个。
            return random.randrange(self.actions)
        else:
            # 利用当前 Q 值最大的动作；并列时选择第一个动作。
            actions_value = [self.table[self._index(a, x, y)] for a in range(self.actions)]
            return actions_value.index(max(actions_value))

    def max_q(self, x, y):
        actions_value = [self.table[self._index(a, x, y)] for a in range(self.actions)]
        return max(actions_value)

    def update(self, a, s0, s1, r, is_terminated, next_action=None):
        """按照 Q-learning 更新公式更新 Q(s, a)。"""
        # s0 和 s1 的格式都是 [x, y]。
        q_predict = self.table[self._index(a, s0[0], s0[1])]
        if not is_terminated:
            # Q-learning 使用下一状态的最大 Q 值，而不是实际执行的下一动作。
            q_target = r + self.gamma * self.max_q(s1[0], s1[1])
        else:
            # 终止状态没有下一步价值，只保留当前奖励。
            q_target = r
        self.table[self._index(a, s0[0], s0[1])] += self.alpha * (q_target - q_predict)