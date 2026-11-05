import random
import time


class Env:
    """Cliff Walking 环境：负责地图状态、动作转移和奖励计算。"""

    def __init__(self, length, height):
        # 定义地图的宽度和高度。
        self.length = length
        self.height = height
        # 智能体从左下角的起点 (0, 0) 出发。
        self.x = 0
        self.y = 0

    def render(self, frames=50):
        # 按行打印当前地图，并用 'o' 标记智能体的位置。
        for i in range(self.height):
            if i == 0:  # 第 0 行是起点、悬崖和终点所在的行。
                # S 表示起点，x 表示悬崖，T 表示终点。
                line = ['S'] + ['x'] * (self.length - 2) + ['T']
            else:
                line = ['.'] * self.length
            if self.x == i:
                line[self.y] = 'o'  # 用 o 覆盖当前位置，表示智能体。
            print(''.join(line))
        # 将光标移回地图顶部，下一帧可以覆盖当前画面形成动画效果。
        print('\033[' + str(self.height + 1) + 'A')
        time.sleep(1.0 / frames)

    def step(self, action):
        """执行一个动作，并返回奖励、下一个状态和终止标志。

        动作编号为：0 表示上，1 表示下，2 表示左，3 表示右。
        """
        # 每个动作对应 (x, y) 坐标的变化量。
        change = [[0, 1], [0, -1], [-1, 0], [1, 0]]

        # 限制坐标范围，防止智能体移动到地图之外。
        self.x = min(self.height - 1, max(0, self.x + change[action][0]))
        self.y = min(self.length - 1, max(0, self.y + change[action][1]))

        states = [self.x, self.y]
        reward = -1  # 每走一步默认获得 -1 的惩罚。
        terminal = False
        if self.x == 0:  # 智能体到达起点、悬崖和终点所在的第 0 行。
            if self.y > 0:  # 不在起点，说明可能进入了悬崖或到达终点。
                terminal = True
                if self.y != self.length - 1:  # 除终点外，第 0 行的位置都属于悬崖。
                    reward = -100  # 掉入悬崖，获得较大的负奖励。
        return reward, states, terminal

    def reset(self):
        # 将环境恢复到初始状态，供新一轮训练使用。
        self.x = 0
        self.y = 0