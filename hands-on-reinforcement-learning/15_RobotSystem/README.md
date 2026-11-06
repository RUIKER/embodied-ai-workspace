# Panda 无渲染冒烟

用 MuJoCo 原生 Python API 加载 `scene.xml`，从 `home` 关键帧重置并验证单关节指令能改变模拟状态。模型来自 [MuJoCo Menagerie 固定提交](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda)，模型的 Apache-2.0 许可证随上游目录保留。

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

本冒烟不使用随机采样。每次从同一 `home` 重置，`joint1` 在 0.4 s 仿真时间内由 0 线性升至 0.1 rad，之后保持到 0.8 s。其余轴保持 `home` 目标，夹爪保持原命令。物理步长为 2 ms，每五步发布一次状态与时钟。稳态窗口为 0.61–0.80 s，预设每轴窗口内平均绝对误差不超过 0.05 rad。采样周期描述仿真时间，墙钟发布率另行测量。

| 检查 | 结果 |
| --- | --- |
| C++ 到 Python | 干净目录 `colcon build` 通过。合法轨迹接收并执行，未知关节、NaN、越界、重复时间点均被拒绝。 |
| 固定轨迹 | `joint1` RMSE 0.01519 rad、峰值误差 0.02418 rad、保持窗口平均绝对误差 0.00119 rad。七轴保持窗口均通过阈值。 |
| 录包与状态回放 | MCAP 包含 5 条命令、80 条关节状态与 80 条时钟消息。停止实时仿真后，状态和时钟逐条回放一致。 |
| 命令重执行 | 新建仿真节点并恢复同一 `home`，只播放命令。新生成的 80 个状态样本中，角度和角速度的最大逐样本差均为 0。 |

每次运行将 `summary.json`、`bag_info.txt`、各进程日志和 MCAP 存到新结果目录。录包使用默认记录时钟，状态误差按消息中的仿真时间对齐。此冒烟只覆盖一条单轴变化轨迹和输入校验，项目正式多轨迹评估、reset 服务、命令停发保护及墙钟频率统计见 [工程验收](../../plan/11_工程主线验收.md)。
