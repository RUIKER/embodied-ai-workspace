"""Pendulum-v1 的解析模型：动力学、线性化、离散化。

为什么自己写一份动力学，而不是直接调 gym 的 env.step：

1. 扰动实验要改质量 m 和长度 l，gymnasium 的 Pendulum-v1 没有这两个构造参数；
2. 延迟实验要让动作晚 k 步生效，env.step 里插不进去；
3. W6 的 C 实现需要一个和被控对象逐项对应的参考实现。

初始状态仍然从 gym 取（reset(seed=...)），保证和 experiments 的评估协议
用的是同一批 20 个起点。自己写的积分器和 gym 是否一致，由
verify_against_gym() 用随机动作逐步对照检查。

======================================================================
一、被控对象
======================================================================
gymnasium 的 Pendulum-v1 是一个均质细杆，一端铰接：

    theta'' = 3g/(2l) * sin(theta) + 3/(m l^2) * u

theta = 0 是直立（不稳定平衡），theta = ±pi 是下垂（稳定平衡）。
u 是力矩，限幅 ±2 N·m；角速度限幅 ±8 rad/s；dt = 0.05 s，每回合 200 步。
单步代价 = angle_normalize(theta)^2 + 0.1*theta_dot^2 + 0.001*u^2，
回报是代价取负后累加。

======================================================================
二、线性化与离散化
======================================================================
在直立点 (theta, theta_dot) = (0, 0) 处，sin(theta) ≈ theta：

    d/dt [theta; theta_dot] = [[0, 1], [3g/(2l), 0]] [theta; theta_dot] + [[0], [3/(m l^2)]] u

记 A、B。LQR 在离散系统上设计，所以还要把 (A, B) 按采样周期 dt 离散化。
这里用 ZOH（零阶保持器）：假设一个周期内力矩不变，被控对象连续演化，
得到

    Ad = e^(A dt),  Bd = ∫_0^dt e^(A s) B ds

两者都可以用矩阵指数一次算出来：把 [A B; 0 0] 乘 dt 后取指数，左上块是 Ad，
右上块是 Bd。这一步在 W6 会和 Euler、Tustin 放在一起对比。
"""

import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are

# Pendulum-v1 的默认参数，与 gymnasium 源码保持一致
G = 10.0
M = 1.0
L = 1.0
DT = 0.05
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
MAX_EPISODE_STEPS = 200


def angle_normalize(x):
    """把角度折到 [-pi, pi)，与 gym 的 angle_normalize 一致。"""
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def step(theta, theta_dot, u, *, m=M, l=L, g=G, dt=DT):
    """积分一步，返回 (theta, theta_dot, cost)。

    参数名 m、l、g 是给扰动实验用的：被控对象可以改成别的质量 / 长度，
    控制器仍然按名义参数设计。
    theta 不折回（与 gym 的 state[0] 一致），折回只在算代价和控制律时做。
    """
    u = float(np.clip(u, -MAX_TORQUE, MAX_TORQUE))
    theta_dot_new = theta_dot + (
        3.0 * g / (2.0 * l) * np.sin(theta) + 3.0 / (m * l**2) * u
    ) * dt
    theta_dot_new = float(np.clip(theta_dot_new, -MAX_SPEED, MAX_SPEED))
    theta_new = theta + theta_dot_new * dt
    cost = angle_normalize(theta) ** 2 + 0.1 * theta_dot**2 + 0.001 * u**2
    return theta_new, theta_dot_new, float(cost)


def continuous_linearization(*, g=G, m=M, l=L):
    """直立点处的连续时间线性化 (A, B)。"""
    A = np.array([[0.0, 1.0], [3.0 * g / (2.0 * l), 0.0]])
    B = np.array([[0.0], [3.0 / (m * l**2)]])
    return A, B


def numerical_linearization(*, g=G, m=M, l=L, eps=1e-6):
    """用中心差分求雅可比，用来核对解析线性化有没有写错。"""

    def f(x, u):
        theta, theta_dot = x
        return np.array(
            [theta_dot, 3.0 * g / (2.0 * l) * np.sin(theta) + 3.0 / (m * l**2) * u]
        )

    x0 = np.zeros(2)
    u0 = 0.0
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    for i in range(2):
        dx = np.zeros(2)
        dx[i] = eps
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2.0 * eps)
    B[:, 0] = (f(x0, u0 + eps) - f(x0, u0 - eps)) / (2.0 * eps)
    return A, B


def zoh_discretize(A, B, dt=DT):
    """零阶保持器离散化，返回 (Ad, Bd)。"""
    n, m = A.shape[0], B.shape[1]
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = A
    aug[:n, n:] = B
    E = expm(aug * dt)
    return E[:n, :n], E[:n, n:]


def lqr_discrete(Ad, Bd, Q, R):
    """离散代数 Riccati 方程的解和最优反馈增益。"""
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def lqr_continuous(A, B, Q, R):
    """连续时间 LQR，只用于和离散解对照。"""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


def verify_against_gym(n_steps=200, seed=0, tol=1e-9):
    """用随机动作逐步对照自写积分器和 gymnasium Pendulum-v1。

    返回最大绝对误差（theta 和 theta_dot 各自）。
    """
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    theta = float(np.arctan2(obs[1], obs[0]))
    theta_dot = float(obs[2])
    rng = np.random.default_rng(seed)

    max_dtheta = 0.0
    max_dtheta_dot = 0.0
    for _ in range(n_steps):
        u = float(rng.uniform(-MAX_TORQUE, MAX_TORQUE))
        theta, theta_dot, _ = step(theta, theta_dot, u)
        obs, _, terminated, truncated, _ = env.step(np.array([u], dtype=np.float32))
        gym_theta = float(np.arctan2(obs[1], obs[0]))
        gym_theta_dot = float(obs[2])
        # theta 可能差 2*pi 的整数倍，比较前先折回
        max_dtheta = max(max_dtheta, abs(angle_normalize(theta - gym_theta)))
        max_dtheta_dot = max(max_dtheta_dot, abs(theta_dot - gym_theta_dot))
        if terminated or truncated:
            break
    env.close()
    return max_dtheta, max_dtheta_dot
