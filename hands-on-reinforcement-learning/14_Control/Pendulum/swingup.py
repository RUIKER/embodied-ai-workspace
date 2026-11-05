"""能量摆起 + LQR 切换：先摆起来，再交给 LQR 稳住。

为什么需要摆起：LQR 是直立点附近的线性控制器，只有起点落在它的吸引域里
才能把杆子拉起来。规划期试算里 LQR 单独只能成功 7/20，失败的全是离直立
较远的起点。摆起的作用是把"离直立很远"变成"进入吸引域"，之后的事情
交回 LQR。

======================================================================
能量控制
======================================================================
均质细杆绕一端转动的机械能（以直立静止为零点）：

    E(theta, theta_dot) = (1/2) I theta_dot^2 + m g (l/2) cos(theta)
    I = m l^2 / 3

直立静止时 E* = m g l / 2，下垂静止时 E = -m g l / 2。
对时间求导（代入被控对象的动力学）后交叉项抵消：

    dE/dt = u * theta_dot

所以取

    u = -k * (E - E*) * sign(theta_dot)

就有 dE/dt = -k (E - E*) |theta_dot|，能量指数收敛到 E*。
theta_dot 恰好为 0 时没有控制力，用一个确定的初始推力打破对称。

======================================================================
切换与滞回
======================================================================
进入 LQR：|theta| < theta_enter 且 |theta_dot| < omega_enter
退出 LQR：|theta| > theta_exit  或 |theta_dot| > omega_exit
退出阈值大于进入阈值（滞回），避免在边界上来回切换造成抖振。

theta_enter / omega_enter / k 在**调参 seed（20000 起）**上网格搜索，
评估 seed（10000–10019）只用来出最终数字，不参与选参。
"""

import numpy as np

from model import G, L, M, MAX_TORQUE, angle_normalize

# 调参 seed 与评估 seed 分开：这是规划期试算里犯过的错，正式版改掉
TUNING_SEEDS = list(range(20000, 20020))
EVAL_SEEDS = list(range(10000, 10020))

DEFAULT_PARAMS = {
    "k": 0.3,
    "theta_enter": 0.5,
    "omega_enter": 2.0,
    "theta_exit": 0.8,
    "omega_exit": 3.0,
}


class LQRController:
    """直立点附近的线性反馈：u = -K [theta, theta_dot]。"""

    def __init__(self, K):
        self.K = np.asarray(K, dtype=float).reshape(2)

    def __call__(self, theta, theta_dot):
        x = np.array([angle_normalize(theta), theta_dot])
        return float(np.clip(-self.K @ x, -MAX_TORQUE, MAX_TORQUE))

    def reset(self):
        pass


class EnergySwingUp:
    """能量控制器，只负责把机械能抬到 E*。"""

    def __init__(self, k=0.3, *, m=M, l=L, g=G):
        self.k = float(k)
        self.I = m * l**2 / 3.0
        self.E_star = m * g * l / 2.0

    def energy(self, theta, theta_dot):
        return 0.5 * self.I * theta_dot**2 + self.E_star * np.cos(theta)

    def __call__(self, theta, theta_dot):
        e = self.energy(theta, theta_dot)
        if abs(theta_dot) < 1e-6:
            # 没有角速度就没有控制力，按"离直立更远的一侧"给一个初始推力
            direction = -1.0 if np.cos(theta) > 0.0 else 1.0
            u = -self.k * (e - self.E_star) * direction
        else:
            u = -self.k * (e - self.E_star) * np.sign(theta_dot)
        return float(np.clip(u, -MAX_TORQUE, MAX_TORQUE))

    def reset(self):
        pass


class SwitchedController:
    """摆起 + LQR，带滞回切换。"""

    def __init__(self, lqr, swing, *, theta_enter, omega_enter, theta_exit, omega_exit):
        self.lqr = lqr
        self.swing = swing
        self.theta_enter = float(theta_enter)
        self.omega_enter = float(omega_enter)
        self.theta_exit = float(theta_exit)
        self.omega_exit = float(omega_exit)
        self.mode = "swing"

    def reset(self):
        self.mode = "swing"

    def __call__(self, theta, theta_dot):
        th = angle_normalize(theta)
        if self.mode == "lqr":
            if abs(th) > self.theta_exit or abs(theta_dot) > self.omega_exit:
                self.mode = "swing"
        else:
            if abs(th) < self.theta_enter and abs(theta_dot) < self.omega_enter:
                self.mode = "lqr"

        if self.mode == "lqr":
            return self.lqr(theta, theta_dot)
        return self.swing(theta, theta_dot)


def build_switched(K, params=None):
    """按参数字典组装切换到 LQR 的控制器。"""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return SwitchedController(
        LQRController(K),
        EnergySwingUp(k=p["k"]),
        theta_enter=p["theta_enter"],
        omega_enter=p["omega_enter"],
        theta_exit=p["theta_exit"],
        omega_exit=p["omega_exit"],
    )
