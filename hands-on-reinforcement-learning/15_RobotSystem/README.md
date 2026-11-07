# Panda 无渲染冒烟

用 MuJoCo 原生 Python API 加载 `scene.xml`，从 `home` 关键帧重置并验证单关节指令能改变模拟状态。模型来自 [MuJoCo Menagerie 固定提交](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda)，模型的 Apache-2.0 许可证随上游目录保留。

学习与排障中的追问和解释见 [经验总结](经验总结.md)。

## 在本机复现

从工作区根目录运行。已有模型缓存和虚拟环境时可直接运行最后一行。

```powershell
$cache = Join-Path $env:LOCALAPPDATA 'EmbodiedAI/PandaSmoke'
$model = Join-Path $cache 'menagerie-98a70c5'
$python = Join-Path $cache '.venv/Scripts/python.exe'
$rev = '98a70c569c4c1b6da467a20012f766ee1f8dc521'
New-Item -ItemType Directory -Force -Path $cache | Out-Null
python -m venv (Join-Path $cache '.venv')
& $python -m pip install 'mujoco==3.17.0' 'numpy==2.5.4'
git clone --filter=blob:none --sparse --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git $model
git -C $model sparse-checkout set franka_emika_panda
git -C $model checkout --detach $rev
& $python -I hands-on-reinforcement-learning/15_RobotSystem/panda_smoke.py "$model/franka_emika_panda/scene.xml"
```

## W2 实测

Windows x64，Python 3.12.10，MuJoCo 3.17.0，NumPy 2.5.4。两次都从同一个 `home` 重置，使用固定输入，测试没有随机采样。每次前 100 步保持 `home` 控制目标，后 150 步将 `joint1` 的目标设为 0.100 rad。其余控制目标不变。

| 检查 | 实测 |
| --- | --- |
| 加载模型 | `nq=9`、`nv=9`、`nu=8`，物理步长 0.002 s |
| 关键帧重置 | 两次重置后 `qpos`、`qvel`、`ctrl`、仿真时间与 `home` 一致 |
| 推进 | 每次 250 步对应 0.500 s 仿真时间，状态有限，无 MuJoCo 警告 |
| 保持阶段 | 七轴相对 `home` 的最大位置偏移 0.00578 rad |
| 单轴阶跃 | `joint1` 结束位置 0.09537 rad，末帧绝对误差 0.00463 rad |
| 同进程重复 | 两条轨迹的 `qpos`、`qvel`、`qacc` 逐点最大绝对差均为 0 |

脚本输出逐轴名称、运行时地址和控制范围。七个臂轴的 `qpos`、`qvel` 地址和执行器下标本次均为 0–6，臂轴位置目标用 rad，臂轴速度用 rad/s。夹爪使用下标 7 的腱执行器，控制值范围 0–255，`home` 为 255。

以上为 Windows 上的 MuJoCo 独立冒烟。WSL 上的集成冒烟见下节。

## ROS2 端到端冒烟

2026-11-06 在 WSL 2 Ubuntu 26.04.1、ROS2 Lyrical、Python 3.14.4、MuJoCo 3.17.0、NumPy 2.5.4 上验证。C++ 发布器发送 `JointTrajectory`，Python 节点校验并执行，在 `/panda_smoke/joint_states` 和 `/clock` 发布数据，rosbag2 写入 MCAP。

### 复现

在 WSL 的 Bash 中运行，先准备 ROS2 Lyrical、`rosbag2`、`trajectory_msgs`、`colcon`、CMake、C++ 编译器和 Python venv。保留上一节的固定版本模型。以下命令从仓库根目录开始，模型目录按本机位置设置。

```bash
source /opt/ros/lyrical/setup.bash
project="$PWD/hands-on-reinforcement-learning/15_RobotSystem"
cache=/mnt/c/Users/PC/AppData/Local/EmbodiedAI/PandaSmoke
scene="$cache/menagerie-98a70c5/franka_emika_panda/scene.xml"
work=$(mktemp -d)
python3 -m venv --system-site-packages "$work/venv"
"$work/venv/bin/python" -m pip install 'mujoco==3.17.0' 'numpy==2.5.4'
cd "$work"
colcon build --base-paths "$project/ros2_smoke/trajectory_publisher" --packages-select panda_trajectory_smoke
source "$work/install/local_setup.bash"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=86
"$work/venv/bin/python" "$project/ros2_smoke/run.py" "$scene" "$cache/ros2-results/$(date +%Y%m%d-%H%M%S)"
```

本机已有可复用的 WSL venv，路径为 `/var/tmp/embodied-panda-ros2-lyrical/ros-venv`。WSL 下载 wheel 超时时，在 Windows 侧从 PyPI 下载 Linux x86_64 / CPython 3.14 的 wheel，再在 WSL 用 `pip install --no-index --find-links "$cache/ros2-wheelhouse"` 安装两个固定版本。加载 ROS 的 Python 节点依赖环境脚本设置的 `PYTHONPATH`。

### 固定输入与结果

下表记录固定目标为 0.1 rad 的历史基线，冒烟发布器源码当前使用 0.15 rad 做参数对照。测试不使用随机采样，每次从同一 `home` 重置，`joint1` 在 0.4 s 内线性升至目标，之后保持到 0.8 s。其余轴保持 `home` 目标，夹爪保持原命令。物理步长为 2 ms，每五步发布一次状态与时钟。稳态窗口为 0.61–0.80 s，预设每轴窗口内平均绝对误差不超过 0.05 rad。采样周期描述仿真时间，墙钟发布率另行测量。

| 检查 | 结果 |
| --- | --- |
| C++ 到 Python | 干净目录 `colcon build` 通过。合法轨迹接收并执行，未知关节、NaN、越界、重复时间点均被拒绝。 |
| 固定轨迹 | `joint1` RMSE 0.01519 rad、峰值误差 0.02418 rad、保持窗口平均绝对误差 0.00119 rad。七轴保持窗口均通过阈值。 |
| 录包与状态回放 | MCAP 包含 5 条命令、80 条关节状态与 80 条时钟消息。停止实时仿真后，状态和时钟逐条回放一致。 |
| 命令重执行 | 新建仿真节点并恢复同一 `home`，只播放命令。新生成的 80 个状态样本中，角度和角速度的最大逐样本差均为 0。 |

每次运行将 `summary.json`、`bag_info.txt`、各进程日志和 MCAP 存到新结果目录。录包使用默认记录时钟，状态误差按消息中的仿真时间对齐。此冒烟只覆盖一条单轴变化轨迹和输入校验，项目正式多轨迹评估、reset 服务、命令停发保护及墙钟频率统计见 [工程验收](../../plan/11_工程主线验收.md)。

## W3 实时关节状态节点

`ros2_ws/src/panda_sim` 提供独立的 `ament_python` 包。节点从 `home` 开始，每次定时回调推进 10 个 MuJoCo 物理步，然后按关节名发布七轴的位置和速度。使用固定模型的 2 ms 物理步长时，配置的发布周期为 20 ms。定时器使用单调时钟，状态消息和 `/clock` 共用步进后的 MuJoCo 时间。

2026-11-06 已在 WSL 2 Ubuntu 26.04.1 / ROS2 Lyrical 的干净目录验证构建与启动，CLI 订阅端收到一条含七轴位置和速度的实时 `JointState`。同一节点订阅 `trajectory`，按下一节的约定执行关节位置轨迹。

在 WSL 终端 A 构建并启动。按本机位置设置仓库路径、模型和已有 MuJoCo venv。

```bash
source /opt/ros/lyrical/setup.bash
project='/mnt/c/Users/PC/Desktop/Embodied_Inteligence_Internship/hands-on-reinforcement-learning/15_RobotSystem'
scene='/mnt/c/Users/PC/AppData/Local/EmbodiedAI/PandaSmoke/menagerie-98a70c5/franka_emika_panda/scene.xml'
panda_python=/var/tmp/embodied-panda-ros2-lyrical/ros-venv/bin/python

cd "$(mktemp -d)"
echo "构建目录 $PWD"
colcon build --base-paths "$project/ros2_ws/src" --packages-select panda_sim panda_trajectory_cmd
source install/local_setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=89
ros2 run --prefix "$panda_python" panda_sim panda_sim_node --ros-args \
  -r __ns:=/panda -p "scene_path:=$scene"
```

在终端 B 检查实时消息。验收信号是 `/panda/panda_sim` 为发布者，消息含 `joint1` 到 `joint7` 及顺序匹配的位置和速度。

```bash
source /opt/ros/lyrical/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=89
ros2 topic info -v /panda/joint_states
ros2 topic echo /panda/joint_states --once --timeout 5
```

保持终端 A 运行。在终端 B 加载 A 打印的构建目录中的 `install/local_setup.bash`，再启动 C++ 发布器。下面的 `build_dir` 要替换为 A 打印的路径。

```bash
build_dir='/tmp/替换为终端A打印的目录名'
source "$build_dir/install/local_setup.bash"
ros2 run panda_trajectory_cmd publish_trajectory --ros-args -r __ns:=/panda
```

C++ 发布器等待发现两个订阅端点，向 `/panda/trajectory` 发送一次固定的两点轨迹，并在执行期间向 `/panda/trajectory_heartbeat` 发心跳。新启动的仿真节点应先记录 `Accepted JointTrajectory with 2 points`，再记录 `Trajectory complete; holding final targets`。此时读取 `/panda/joint_states`，可观察 `joint1` 接近 0.1 rad。

## W4 固定步长轨迹执行

- 轨迹必须包含七个不同的臂轴名，节点按名称将每个点重排到模型顺序。支持位置字段。点值须有限，并落在关节范围和执行器控制范围的交集内。
- `header.stamp` 须为零。第一个 `time_from_start` 须为零，之后严格递增。第一个点须与当前保持目标一致，绝对容差为 1 µrad。执行中收到新命令记录 `busy` 并拒绝。
- 接收时记录当前 MuJoCo 时间。每个物理步开始前按该时间差做分段线性插值，将结果写到臂轴的 `ctrl`，再推进物理。经过末点时保持最后一组位置目标。C++ 示例从 `home` 出发，重跑同一示例前可重启仿真节点。

轨迹执行功能检查使用固定输入，无随机采样。接入心跳前，已验证 C++ 示例在 0.4 s 内将 `joint1` 从 0 指向 0.1 rad，实测角进入目标 ±0.005 rad。关节名倒序的回程轨迹在 0.6 s 内回到初始目标，并用重复命令验证了 `busy`。错误关节名、非零时间戳、点维度错误、重复时间点、NaN、越界和首点不连续均通过拒绝检查。

### 仿真时钟重置与停发

- `/clock` 和 `JointState.header.stamp` 取同一个仿真采样时刻。每个 ROS 域使用一个 `/clock` 来源。用仿真时间的下游节点启用 `use_sim_time`，等到首个非零时钟再处理数据。
- `/panda/reset` 使用 `std_srvs/srv/Trigger`。执行中返回失败，空闲时恢复 `home` 的状态和控制量，并重建 volatile 轨迹订阅以清除其积压。下一帧时钟从新的仿真起点推进，录包与离线分析按回退分段。
- 单发送方约定下，C++ 示例每隔 50 ms 发一次空心跳。接收端用单调时间检查，活动轨迹在 250 ms 内未收到心跳时终止插值，保持最后已应用的 `ctrl`。迟到的心跳不会恢复已超时的轨迹。轨迹正常结束后保持末点。`/clock` 和心跳使用 best effort、队列深度 1，轨迹命令使用 reliable。

终端 A 运行仿真时，可在终端 B 观察时钟、调用 reset。执行完 C++ 示例后重置，可从 `home` 再次运行同一示例。

```bash
ros2 topic echo /clock --qos-reliability best_effort --once --timeout 5
ros2 service call /panda/reset std_srvs/srv/Trigger '{}'
```

要在空闲重置后复现无心跳超时，可单独发送一条两秒轨迹。终端 A 应记录 `Trajectory heartbeat timed out; holding last targets`，随后 `joint1` 保持在中间位置附近。

```bash
ros2 topic pub --once /panda/trajectory trajectory_msgs/msg/JointTrajectory \
  '{joint_names: [joint1, joint2, joint3, joint4, joint5, joint6, joint7], points: [{positions: [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]}, {positions: [0.5, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], time_from_start: {sec: 2}}]}'
```

WSL 功能检查验证了状态与时钟的时间戳一致、仿真时钟消费、忙时拒绝重置、空闲重置后时钟回退、运行中心跳停发后保持，以及 reset 后重新执行 C++ 示例。停发场景使用上面的两秒轨迹，按 50 ms 间隔发送五次心跳后停发，并检查迟到的心跳未恢复运动。原有冒烟和录包回放复跑通过。正式录包时把心跳和轨迹一起视为命令流，重执行时保留实时仿真发布的 `/clock`。

## W5 正式节点录包与回放冒烟

使用上述 `panda_sim` 和 `panda_trajectory_cmd` 正式节点。C++ 发一条固定轨迹，`joint1` 在 0.4 s 内从 0 变到 0.1 rad，其余轴保持 `home` 目标，无随机采样。录包保存 `/panda/trajectory`、`/panda/trajectory_heartbeat`、`/panda/joint_states` 和 `/clock`。录包以接收时间保留消息顺序与心跳间隔，轨迹对比用 `JointState` 自带的仿真时间。

1. 关闭实时节点，只从 MCAP 播放 `/panda/joint_states` 和 `/clock`。检查它们均来自播放器，关节名、位置、速度和消息时间戳逐条与包内数据一致。录制时 `/clock` 为 best effort，回放检查临时用 reliable 逐条比对。
2. 重新启动仿真节点，从 `home` 复位，再只播放轨迹和心跳。实时仿真是 `/clock` 和状态的唯一来源。两次都从各自接受命令的仿真时刻开始对齐，比较后续 20–800 ms 的 40 帧七轴位置和速度，每帧都核对对应的 `/clock` 时间戳。冒烟要求两次轨迹均正常完成，且 `joint1` 的末帧位置进入目标 ±0.005 rad。两个进程接收命令前保持 `home` 的时长可能不同，报告实测差异。

从仓库根目录在 WSL Bash 运行。沿用上文的固定版本模型和 ROS2 Lyrical venv，另需 `rosbag2_py` 和 MCAP 存储插件。为测试使用空闲的本地 ROS 域，避免其他节点也发布 `/clock`。

```bash
source /opt/ros/lyrical/setup.bash
project="$PWD/hands-on-reinforcement-learning/15_RobotSystem"
cache=/mnt/c/Users/PC/AppData/Local/EmbodiedAI/PandaSmoke
scene="$cache/menagerie-98a70c5/franka_emika_panda/scene.xml"
panda_python=/var/tmp/embodied-panda-ros2-lyrical/ros-venv/bin/python

cd "$(mktemp -d)"
colcon build --base-paths "$project/ros2_ws/src" --packages-select panda_sim panda_trajectory_cmd
source install/local_setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=106
"$panda_python" "$project/ros2_smoke/run_formal.py" "$scene" \
  "$cache/ros2-results/$(date +%Y%m%d-%H%M%S)-formal-bag"
```

每次运行保存 `summary.json`、`bag_info.txt`、进程日志、QoS 配置和 MCAP。2026-11-07 的 `20261107-092518-formal-bag` 在 WSL 2 Ubuntu 26.04.1、ROS2 Lyrical、MuJoCo 3.17.0、NumPy 2.5.4 上通过。录得 1 条轨迹、12 条心跳、94 条状态与 90 条时钟，状态与时钟各自完整回放。重执行的 40 帧中，七轴位置和速度的最大绝对差分别为 `8.08e-9 rad`、`8.78e-8 rad/s`，两次 `joint1` 末帧均约为 `0.099622 rad`。整包的状态和时钟计数可以不同，比较窗口内逐帧检查二者的时间戳。rosbag2 对空心跳类型可能提示缺少内嵌消息定义，脚本使用本机 `std_msgs/msg/Empty` 验证解码和回放。
