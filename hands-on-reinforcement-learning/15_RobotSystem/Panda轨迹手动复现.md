# Panda 固定轨迹手动复现

对应 11 月 9 日的个人练习。用终端 A 运行仿真并观察接收日志，用终端 B 读取状态、发命令和调用 reset。两端都使用本地 ROS 域 118。正常样本固定为单轴轨迹，从 `home` 出发，`joint1` 在 0.4 s 内的目标从 0 增加到 0.1 rad，其余六轴目标保持 `home`。

## 终端 A 构建并启动

在 Ubuntu WSL 的 Bash 中运行。将仓库路径、固定模型和 venv 路径按本机位置设置。

```bash
source /opt/ros/lyrical/setup.bash
printf 'ROS_DISTRO=%s\n' "$ROS_DISTRO"
command -v colcon

project='/mnt/c/Users/PC/Desktop/Embodied_Inteligence_Internship/hands-on-reinforcement-learning/15_RobotSystem'
scene='/mnt/c/Users/PC/AppData/Local/EmbodiedAI/PandaSmoke/menagerie-98a70c5/franka_emika_panda/scene.xml'
panda_python='/var/tmp/embodied-panda-ros2-lyrical/ros-venv/bin/python'
build_dir=$(mktemp -d /tmp/panda-nov09-manual.XXXXXX)
cd "$build_dir" &&
  colcon build --base-paths "$project/ros2_ws/src" --packages-select panda_sim panda_trajectory_cmd &&
  printf 'BUILD_DIR=%s\n' "$build_dir"
```

确认编译完成后，在 A 启动仿真并保持运行。设置 `send_demo:=false` 后，可从 B 手动决定何时发送轨迹。

```bash
source "$build_dir/install/local_setup.bash"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=118
ros2 launch panda_sim panda_demo.launch.py \
  "scene_path:=$scene" "sim_python:=$panda_python" send_demo:=false
```

## 终端 B 读取状态并发送轨迹

启动日志显示 `Publishing Panda joint states` 后，在另一个 Ubuntu WSL 终端运行以下命令。

```bash
source /opt/ros/lyrical/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=118
ros2 topic echo /panda/joint_states sensor_msgs/msg/JointState --once --timeout 5
ros2 topic echo /clock rosgraph_msgs/msg/Clock --qos-reliability best_effort --once --timeout 5
```

`name[i]`、`position[i]` 和 `velocity[i]` 分别是同一关节的名称、实测位置和速度。每次物理推进后，节点给当次状态和时钟消息填同一个仿真采样时刻。两条 `--once` 命令分别取下一帧，所以两次读取的时间戳通常不同。

把下面的 `build_dir` 改为 A 实际输出的路径。发送后，在 A 确认 `Accepted JointTrajectory with 2 points` 和 `Trajectory complete; holding final targets`，再读一次状态。预期 `joint1` 进入 0.1 rad ± 0.005 rad。

```bash
build_dir='/tmp/替换为终端A输出的构建目录名'
source "$build_dir/install/local_setup.bash"
ros2 run panda_trajectory_cmd publish_trajectory \
  --ros-args -r __ns:=/panda -p trajectory_case:=single_axis
ros2 topic echo /panda/joint_states sensor_msgs/msg/JointState --once --timeout 5
```

## 复现拒绝并恢复

等正常轨迹结束后，先预测 `joint_typo` 会触发哪一层检查，以及 `joint1` 会不会向 0.2 rad 移动。然后在 B 发送错误样本。

```bash
ros2 topic pub --once /panda/trajectory trajectory_msgs/msg/JointTrajectory \
  '{joint_names: [joint_typo, joint2, joint3, joint4, joint5, joint6, joint7], points: [{positions: [0.1, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]}, {positions: [0.2, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], time_from_start: {nanosec: 400000000}}]}'
ros2 topic echo /panda/joint_states sensor_msgs/msg/JointState --once --timeout 5
```

A 应显示 `Rejected JointTrajectory: joint_names`，`joint1` 仍接近 0.1 rad。接收节点先校验关节名，再读取轨迹点和检查首点，所以该错误发生在轨迹执行之前。

在 B 调用 reset 并重发正常轨迹。reset 恢复 `home` 保持目标，重新满足示例轨迹首点条件。

```bash
ros2 service call /panda/reset std_srvs/srv/Trigger '{}'
ros2 topic echo /panda/joint_states sensor_msgs/msg/JointState --once --timeout 5
ros2 run panda_trajectory_cmd publish_trajectory \
  --ros-args -r __ns:=/panda -p trajectory_case:=single_axis
ros2 topic echo /panda/joint_states sensor_msgs/msg/JointState --once --timeout 5
```

记录 reset 的 `success=True`、恢复后的接受和完成日志，以及 reset 前后和重跑后的 `joint1` 位置。练习结束后在 A 按 Ctrl+C 停止仿真。
