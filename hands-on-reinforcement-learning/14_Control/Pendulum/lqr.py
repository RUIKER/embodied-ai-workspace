"""线性化 + 离散 LQR：把公式落到一个可以检查的增益上。

跑法：
    python -I 14_Control/Pendulum/lqr.py

输出五件事：
1. 解析线性化和数值差分的 A、B 是否一致（防止手推写错符号）；
2. ZOH 离散化后的 Ad、Bd；
3. Q = diag(1, 0.1)、R = 0.001 下的 K（离散与连续各算一遍）；
4. 闭环极点 lambda(Ad - Bd K) 是否都在单位圆内；
5. 名义参数下，不同 Q/R 取值对 K 的影响（一小组，不是调参）。

Q、R 是设计者的取舍，不是标准答案：Q 大 -> 更在意角度误差；
R 大 -> 更不愿意用力矩。这里选 Q = diag(1, 0.1)、R = 0.001，
是因为 Pendulum 的力矩上限只有 2，希望角度误差收敛得快一些。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from model import (
    DT,
    continuous_linearization,
    lqr_continuous,
    lqr_discrete,
    numerical_linearization,
    zoh_discretize,
)


def closed_loop_poles(Ad, Bd, K):
    return np.linalg.eigvals(Ad - Bd @ K)


def main():
    parser = argparse.ArgumentParser(description="Pendulum 线性化与 LQR 设计")
    parser.add_argument("--q-theta", type=float, default=1.0, help="Q 的角度项")
    parser.add_argument("--q-theta-dot", type=float, default=0.1, help="Q 的角速度项")
    parser.add_argument("--r", type=float, default=0.001, help="R 的力矩项")
    parser.add_argument("--dt", type=float, default=DT)
    args = parser.parse_args()

    print("=" * 64)
    print("1. 线性化（直立点 theta = 0）")
    print("=" * 64)
    A, B = continuous_linearization()
    A_num, B_num = numerical_linearization()
    print("A (解析):")
    print(A)
    print("B (解析):", B.ravel())
    print("A (数值差分):")
    print(A_num)
    print("B (数值差分):", B_num.ravel())
    print(f"最大偏差: A {np.max(np.abs(A - A_num)):.3e}, B {np.max(np.abs(B - B_num)):.3e}")

    print()
    print("=" * 64)
    print(f"2. ZOH 离散化（dt = {args.dt} s）")
    print("=" * 64)
    Ad, Bd = zoh_discretize(A, B, args.dt)
    print("Ad:")
    print(Ad)
    print("Bd:", Bd.ravel())

    print()
    print("=" * 64)
    print("3. LQR 增益")
    print("=" * 64)
    Q = np.diag([args.q_theta, args.q_theta_dot])
    R = np.array([[args.r]])
    K_d, P_d = lqr_discrete(Ad, Bd, Q, R)
    K_c, P_c = lqr_continuous(A, B, Q, R)
    print(f"Q = diag({args.q_theta}, {args.q_theta_dot}), R = {args.r}")
    print(f"K (离散, ZOH): {K_d.ravel()}")
    print(f"K (连续):       {K_c.ravel()}")
    print(f"参考值（规划期试算）: [19.7, 5.26]")

    print()
    print("=" * 64)
    print("4. 闭环极点")
    print("=" * 64)
    poles = closed_loop_poles(Ad, Bd, K_d)
    print("lambda(Ad - Bd K):", np.round(poles, 6))
    print(f"模长: {np.round(np.abs(poles), 6)}")
    stable = bool(np.all(np.abs(poles) < 1.0))
    print(f"全部在单位圆内: {stable}")

    print()
    print("=" * 64)
    print("5. Q / R 对 K 的影响（同一组名义参数）")
    print("=" * 64)
    print(f"{'Q_theta':>8} {'Q_omega':>8} {'R':>8}   {'K_theta':>10} {'K_omega':>10}")
    for q_theta in (0.1, 1.0, 10.0):
        for q_omega in (0.01, 0.1, 1.0):
            for r in (0.01, 0.001, 0.0001):
                K, _ = lqr_discrete(Ad, Bd, np.diag([q_theta, q_omega]), np.array([[r]]))
                print(
                    f"{q_theta:>8.2f} {q_omega:>8.2f} {r:>8.4f}   "
                    f"{K.ravel()[0]:>10.3f} {K.ravel()[1]:>10.3f}"
                )


if __name__ == "__main__":
    main()
