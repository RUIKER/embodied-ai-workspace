"""回合循环与指标，由 evaluate / tune / robustness 共用。

指标口径（写在前面，避免事后改定义）：

- 固定起点：与 experiments/seed_experiments/five_algorithms 完全相同，
  从 gym 的 reset(seed=10000+i) 取初始状态，20 个起点全算法共用。
- 成功（严格）：最后 2.5 s（50 步）每一步 |angle_normalize(theta)| < 0.15 rad。
  这是本项目的定义，不是 gym 或论文的标准。
- 成功（回报）：回合回报 >= -200。experiments 的 -200 是对 20 个起点
  的平均回报说的；这里借来做逐回合的粗判据，两套都报，不混用。
- 摆起时间：第一个"此后每一步都满足 |theta| < 0.15"的时刻；从未进入则记 None。
"""

import gymnasium as gym
import numpy as np

from model import DT, MAX_EPISODE_STEPS, MAX_TORQUE, angle_normalize, step

SUCCESS_ANGLE = 0.15
SUCCESS_WINDOW = int(round(2.5 / DT))  # 50 步
RETURN_THRESHOLD = -200.0


def initial_states(seeds):
    """用 gym 的 reset 取初始状态，保证与 experiments 的评估起点一致。"""
    env = gym.make("Pendulum-v1")
    states = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        states.append((float(np.arctan2(obs[1], obs[0])), float(obs[2])))
    env.close()
    return states


def rollout(
    controller,
    theta0,
    theta_dot0,
    *,
    m=1.0,
    l=1.0,
    g=10.0,
    delay_steps=0,
    max_steps=MAX_EPISODE_STEPS,
):
    """跑一个回合，返回逐回合指标。"""
    if hasattr(controller, "reset"):
        controller.reset()

    theta, theta_dot = float(theta0), float(theta_dot0)
    delay_queue = [0.0] * delay_steps  # 动作延迟：晚 k 步生效，初始动作为 0
    total_return = 0.0
    angles = []

    for _ in range(max_steps):
        u_cmd = float(np.clip(controller(theta, theta_dot), -MAX_TORQUE, MAX_TORQUE))
        if delay_steps > 0:
            u_applied = delay_queue.pop(0)
            delay_queue.append(u_cmd)
        else:
            u_applied = u_cmd
        theta, theta_dot, cost = step(theta, theta_dot, u_applied, m=m, l=l, g=g)
        total_return -= cost
        angles.append(angle_normalize(theta))

    angles = np.asarray(angles)
    tail = angles[-SUCCESS_WINDOW:]

    stable_from = None
    for t in range(len(angles)):
        if np.all(np.abs(angles[t:]) < SUCCESS_ANGLE):
            stable_from = t
            break

    return {
        "return": float(total_return),
        "success_strict": bool(np.all(np.abs(tail) < SUCCESS_ANGLE)),
        "success_return": bool(total_return >= RETURN_THRESHOLD),
        "time_to_upright": None if stable_from is None else float(stable_from * DT),
        "final_angle": float(angles[-1]),
        "max_abs_angle_tail": float(np.max(np.abs(tail))),
    }


def summarize(rows):
    """把逐回合结果聚合成一行汇总。"""
    returns = np.array([r["return"] for r in rows], dtype=float)
    times = [r["time_to_upright"] for r in rows if r["time_to_upright"] is not None]
    return {
        "n": len(rows),
        "success_strict_rate": float(np.mean([r["success_strict"] for r in rows])),
        "success_return_rate": float(np.mean([r["success_return"] for r in rows])),
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "mean_time_to_upright": float(np.mean(times)) if times else None,
    }


def evaluate_controller(controller, seeds, states=None, **plant_kwargs):
    """在一组固定起点上跑控制器，返回逐回合结果和汇总。

    states 可以预先算好传进来（调参时 100 组参数共用同一批起点，
    不必每组都重建一次 gym 环境）。
    """
    if states is None:
        states = initial_states(seeds)
    rows = []
    for seed, (theta0, theta_dot0) in zip(seeds, states):
        row = rollout(controller, theta0, theta_dot0, **plant_kwargs)
        row["seed"] = int(seed)
        rows.append(row)
    return rows, summarize(rows)
