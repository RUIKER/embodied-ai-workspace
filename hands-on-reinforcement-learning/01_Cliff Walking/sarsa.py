import random

from config import EXPLORATION_RATE


class SarsaTable:
	"""SARSA 的 Q 表、动作选择和参数更新逻辑。"""

	def __init__(self, length, height, actions=4, alpha=0.1, gamma=0.9):
		# 将 Q(s, a) 表展平成一维列表，所有状态-动作价值初始为 0。
		self.table = [0] * actions * length * height
		self.actions = actions
		self.length = length
		self.height = height
		self.alpha = alpha
		self.gamma = gamma

	def _index(self, action, x, y):
		"""返回 Q([x, y], action) 在一维 Q 表中的下标。"""
		return action * self.height * self.length + x * self.length + y

	def _epsilon(self):
		"""返回 epsilon-greedy 策略中的随机探索概率。"""
		return EXPLORATION_RATE

	def take_action(self, x, y):
		"""使用 epsilon-greedy 策略选择动作。"""
		if random.random() < self._epsilon():
			# 探索：随机选择一个动作。
			return random.randrange(self.actions)

		# 利用：选择当前状态下 Q 值最大的动作。
		action_values = [
			self.table[self._index(action, x, y)]
			for action in range(self.actions)
		]
		return action_values.index(max(action_values))

	def update(self, action, state, next_state, reward, is_terminated, next_action=None):
		"""按照 SARSA 公式更新 Q(s, a)。

		SARSA 使用实际执行的下一动作 next_action 计算目标值，
		而不是像 Q-learning 那样选择下一状态的最大 Q 值。
		"""
		current_index = self._index(action, state[0], state[1])
		q_predict = self.table[current_index]

		if is_terminated:
			# 终止状态没有下一步动作，因此目标值只有当前奖励。
			q_target = reward
		else:
			if next_action is None:
				raise ValueError("非终止状态必须提供 next_action。")
			next_index = self._index(next_action, next_state[0], next_state[1])
			q_target = reward + self.gamma * self.table[next_index]

		self.table[current_index] += self.alpha * (q_target - q_predict)
