import numpy as np


# 固定关节 O 的位置，两根杆的长度
BASE = np.array([-1.0, 0.0])
L1 = 1.0
L2 = 1.0


def forward_kinematics(q):
    """输入关节角 q = [q1, q2]，输出末端位置 [x, y]。角度单位：弧度。"""
    q1, q2 = q

    x = BASE[0] + L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = BASE[1] + L1 * np.sin(q1) + L2 * np.sin(q1 + q2)

    return np.array([x, y])


def jacobian(q):
    """计算当前位置对应的位置雅可比矩阵，形状为 2×2。"""
    q1, q2 = q

    return np.array([
        [
            -L1 * np.sin(q1) - L2 * np.sin(q1 + q2),
            -L2 * np.sin(q1 + q2),
        ],
        [
            L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
            L2 * np.cos(q1 + q2),
        ],
    ])


def solve_ik(
    target,
    q_initial,
    damping=1.0,
    tolerance=1e-4,
    max_iterations=100,
):
    """
    使用阻尼最小二乘法求解末端位置 IK。

    返回：
        q：最终关节角，单位 rad
        success：是否达到位置误差要求
        iterations：已执行的关节角更新次数
    """
    if damping <= 0:
        raise ValueError("damping 必须大于 0")

    target = np.asarray(target, dtype=float)
    q = np.asarray(q_initial, dtype=float).copy()

    # k 表示已执行的关节角更新次数，包括初始状态 k=0
    for k in range(max_iterations + 1):
        position = forward_kinematics(q)
        error = target - position
        error_norm = np.linalg.norm(error)

        print(
            f"更新次数={k:3d} | "
            f"q={np.round(np.degrees(q), 3)} deg | "
            f"末端={np.round(position, 5)} m | "
            f"误差={error_norm:.6f} m"
        )

        if error_norm < tolerance:
            return q, True, k

        if k == max_iterations:
            break

        J = jacobian(q)

        # 等价于：
        # dq = J.T @ inverse(J @ J.T + damping**2 * I) @ error
        # solve 直接解线性方程，避免显式求逆。
        matrix = J @ J.T + damping**2 * np.eye(2)
        dq = J.T @ np.linalg.solve(matrix, error)

        # 教学示例：假设关节精确执行了整步 dq。
        q = q + dq

    return q, False, max_iterations


if __name__ == "__main__":
    target = np.array([0.0, 1.1])
    q_initial = np.radians([0.0, 90.0])

    q_result, success, iterations = solve_ik(target, q_initial)

    print("\n求解成功：", success)
    print("更新次数：", iterations)
    print("最终关节角：", np.degrees(q_result), "deg")
    print("最终末端位置：", forward_kinematics(q_result), "m")