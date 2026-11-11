# Panda 无渲染冒烟

用 MuJoCo 原生 Python API 加载 `scene.xml`，从 `home` 关键帧重置并验证单关节指令能改变模拟状态。模型来自 [MuJoCo Menagerie 固定提交](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda)，模型的 Apache-2.0 许可证随上游目录保留。

学习与排障中的追问和解释见 [经验总结](经验总结.md)。

11 月 9 日分终端练习见 [Panda 固定轨迹手动复现](Panda轨迹手动复现.md)。

## 额外练习：关节阻尼受控对照

`panda_damping_compare.py` 加载同一固定模型，以 `home` 关键帧重置。基线使用原始 `joint1` 关节阻尼，对照组只将该轴的 `dof_damping` 乘以指定倍数。两组其余模型参数、初始状态、执行器控制和固定输入一致。脚本直接使用 MuJoCo 推进仿真，命令轨迹复用单轴示例的目标值和线性插值规则。

无随机采样。每组分别从同一关键帧运行两次，每次从 0 rad 目标在 0.4 s 内线性到 0.1 rad，再保持到 0.8 s。物理步长 0.002 s。每步先设置位置目标，再推进一步；记录该步开始的控制时间、实际施加的目标与推进后的状态时间和位置。跟踪 RMSE 比较每步末位置和该步施加目标。

在 WSL 中用已有模型和 Python 环境运行。输出目录必须为新目录。

```bash
source /opt/ros/lyrical/setup.bash
project='/mnt/c/Users/PC/Desktop/Embodied_Inteligence_Internship/hands-on-reinforcement-learning/15_RobotSystem'
scene='/mnt/c/Users/PC/AppData/Local/EmbodiedAI/PandaSmoke/menagerie-98a70c5/franka_emika_panda/scene.xml'
panda_python='/var/tmp/embodied-panda-ros2-lyrical/ros-venv/bin/python'
out="/mnt/c/Users/PC/Documents/ChatGPT/EIP/damping-$(date +%Y%m%d-%H%M%S)"
"$panda_python" "$project/panda_damping_compare.py" "$scene" "$out" --damping-factor 2
```

输出 `trajectory.csv` 和 `summary.json`。同组两次位置差用于检查固定条件下的重复性。对照组第一次减基线第一次的位置差用于观察阻尼变化下的响应差异。脚本同时报告整段跟踪 RMSE、末帧位置和四个时间点的样本。这个离线练习不包含 ROS2 通信或传感器噪声。

Agent 在 WSL 2、MuJoCo 3.17.0、NumPy 2.5.4 上用 `--damping-factor 1` 检查对照两侧完全一致，再用 `--damping-factor 2` 检查阻尼变化产生可观察差异。本人于 2026-11-08 在相同版本下运行双倍阻尼对照，原始数据、摘要与图分别保存在 [轨迹数据](results/20261108_joint1_damping/trajectory.csv)、[摘要](results/20261108_joint1_damping/summary.json) 和 [对照图](results/20261108_joint1_damping/comparison.png)。

使用带 NumPy 和 Matplotlib 的 Python，从某次实验目录生成图。

```bash
python "$project/panda_damping_plot.py" "$out"
```

输出 `comparison.png` 包括关节位置、两组位置差、速度、阻尼力矩分量、跟踪误差和组内重执行差异。阻尼力矩分量按采样时的 `−dof_damping × qvel` 重建，单位 N·m；执行器力矩没有包含在该量内。图中目标值是在物理步开始时施加的控制目标，状态值取该步结束时的采样。WSL 现有 MuJoCo venv 是否安装 Matplotlib 需单独确认；本次绘图使用 Windows 的独立临时 Python 依赖。

## Panda 正运动学检查

`panda_fk.py` 以模型的 `link0` 为参考坐标系，计算 `hand` 坐标系的位姿。末端选择的是模型中 `hand` 的原点，夹爪指尖或工具中心需要额外的坐标变换。

脚本按名称找到 `link0` 到 `hand` 的连杆与七个关节，从编译后的模型读取每段固定位置、固定旋转、局部关节轴及关节位置，再逐段相乘。计算过程独立组合变换，模型参数来自同一份 MuJoCo 模型。对照值来自 `mj_forward` 后的 body 位姿，两者都转换到 `link0` 坐标系。

在 WSL 中使用已有模型和 MuJoCo 环境运行。

```bash
source /opt/ros/lyrical/setup.bash
project='/mnt/c/Users/PC/Desktop/Embodied_Inteligence_Internship/hands-on-reinforcement-learning/15_RobotSystem'
scene='/mnt/c/Users/PC/AppData/Local/EmbodiedAI/PandaSmoke/menagerie-98a70c5/franka_emika_panda/scene.xml'
panda_python='/var/tmp/embodied-panda-ros2-lyrical/ros-venv/bin/python'

"$panda_python" "$project/panda_fk.py" "$scene"
"$panda_python" "$project/panda_fk.py" "$scene" \
  --joints 0.1 -0.1 0 -1.57079 0 1.57079 -0.7853
```

默认固定输入共九组，不涉及随机采样。包括 `home`、第一轴加 0.1 rad 且第二轴减 0.1 rad，以及每次仅一轴相对 `home` 增加 0.05 rad 的七组。脚本输出两种位置和 3×3 旋转矩阵，比较位置欧氏误差和旋转矩阵逐元素最大绝对差，二者各自超过 `1e-9` 时退出失败。参数 `--joints` 接收七个有限关节角，单位为 rad。

2026-11-08 在 WSL 2、MuJoCo 3.17.0、NumPy 2.5.4 上完成上述九组 agent 检查，最大位置差为 `4.710e-16 m`，最大旋转矩阵元素差为 `2.665e-15`。对照检查反映同模型下的数值一致性，精度受到模型及输入定义影响。本人手动复现见 [W3 清单](../../plan/02_W3部署清单.md)。

## Panda 两轴到达任务

`panda_two_joint_env.py` 提供原生 Python 的 `reset()` 和 `step(action)`。`reset()` 恢复 `home`，默认任务目标为 `(0.1, 0.1)` rad。观测依次是 `q1, q2, dq1, dq2, goal1, goal2, ctrl1, ctrl2`。动作是两轴位置命令增量，每轴限于 ±0.02 rad，累加后的命令限制在模型关节和执行器允许范围的交集。其他执行器保留 home 命令。每个动作推进 10 个 2 ms 物理步。

两个受控关节的位置误差都小于 0.01 rad 且速度绝对值都小于 0.05 rad/s 时成功，奖励为 0。其他步奖励为 −1。无效状态、仿真警告或受控关节超出模型限位及 0.001 rad 容差时终止为失败。默认 250 步无成功或失败时记为超时，返回 `truncated=True`。这些是固定场景 smoke 的初始参数。

在 WSL 中沿用上文的 `project`、`scene` 和 `panda_python`，先预测命令目标再运行。

```bash
"$panda_python" "$project/panda_two_joint_env.py" "$scene" \
  --action 0.02 -0.01 --action 0.02 0.02
```

每行是一次 `reset` 或 `step` 的 JSON。两步执行器目标预期分别为 `(0.02, -0.01)` 和 `(0.04, 0.01)` rad。可用 `--goal 0.1 0.05` 改变任务目标后复跑。

2026-11-08，agent 在固定模型、MuJoCo 3.17.0、NumPy 2.5.4 下验证复位与重复执行、命令及增量限幅、非法动作、超时和注入限位异常。成功分支用固定目标，先连续五步施加 `(0.02, 0.02)` rad，再施加零增量直至成功，于第 18 步通过。全部短检查无随机采样。本人于 2026-11-09 在 WSL 完成两步固定动作复现。

<details>
<summary>两步手动复现参考值</summary>

2026-11-09，agent 用上面的命令在 WSL 2 和 Windows 上复跑固定输入。两边均使用 MuJoCo 3.17.0、NumPy 2.5.4 和页首固定版本的模型，无随机采样。下表保留六位小数。

| 阶段 | 实际角度 q（rad） | 实际角速度 dq（rad/s） | 执行器目标 ctrl（rad） | 奖励 |
| --- | --- | --- | --- | --- |
| reset | `(0, 0)` | `(0, 0)` | `(0, 0)` | — |
| 第一步 | `(0.003600, -0.000598)` | `(0.166862, -0.029388)` | `(0.02, -0.01)` | `-1` |
| 第二步 | `(0.010277, 0.001637)` | `(0.307635, 0.142928)` | `(0.04, 0.01)` | `-1` |

三行观测的任务目标均为 `(0.1, 0.1)` rad，两步均为 `running`，`terminated` 与 `truncated` 都为 `false`。在 Windows 上加入 `--goal 0.1 0.05` 并保持两步动作不变，三行中只有观测的 `goal2` 从 `0.1` 变为 `0.05`。任务目标参与到达判定，固定动作通过 `ctrl` 驱动仿真。本人在 WSL 手算并复现了两组目标下的两步输出。

</details>

## Panda 两轴 SAC CPU 检查

`panda_sac_smoke.py` 复用 `11_SAC/Pendulum` 的 SAC 和回放池。策略和回放使用归一化动作 `[-1, 1]²`，送入环境时乘以 0.02，转换为关节目标增量。回合在终止或超时后重置。回放池只把 `terminated` 作为停止 bootstrap 的标记。

11 月 10 日的固定动作练习、源码追踪及正常步/超时步对照见 [SAC 单条样本追踪](SAC单条样本追踪.md)。Agent 已验证该入口；本人复现和口述待完成。

从项目根目录在 Windows PowerShell 运行。使用 Python 3.12 和隔离的 CPU 环境。已有环境可跳过安装。

```powershell
$cpuEnv = 'C:\Users\PC\Documents\ChatGPT\EIP\.venv-panda-sac-cpu'
python -m venv $cpuEnv
$pandaPython = Join-Path $cpuEnv 'Scripts\python.exe'
& $pandaPython -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu'
& $pandaPython -m pip install 'mujoco==3.17.0' 'numpy==2.5.4'
$scene = Join-Path $env:LOCALAPPDATA 'EmbodiedAI\PandaSmoke\menagerie-98a70c5\franka_emika_panda\scene.xml'
$out = Join-Path (Split-Path $cpuEnv) ('panda-sac-cpu-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
& $pandaPython hands-on-reinforcement-learning/15_RobotSystem/panda_sac_smoke.py $scene $out
```

协议固定 seed 42、64 条采样、32 条 batch、一次参数更新。测试将单回合临时限为 16 步，用于覆盖超时重置。输出 `summary.json` 和 `sac_cpu.pt`，其中 checkpoint 包含网络、优化器、温度参数和配置。以相同状态和随机种子对照重载前后的动作。该流程验收接线、更新和保存重载，策略成功率另行评估。

2026-11-08 在 Windows CPU、PyTorch 2.10.0+cpu、MuJoCo 3.17.0、NumPy 2.5.4 上通过。Actor 和两个 Critic 的最大参数变化均约 `3.0e-4`，重载动作最大绝对差为 `0`。

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

录包脚本将 C++ 发布器的 `required_subscribers` 设为 2，等仿真节点和录包器分别匹配轨迹及心跳话题后才发送。直接运行发布器和 launch 时默认只要求 1 个订阅者。

## 正式启动与参数

沿用上一节的 WSL 环境，在干净目录构建两个包并加载 overlay。使用空闲 ROS 域，每个域只运行一个发布 `/clock` 的仿真节点。

```bash
ros2 launch panda_sim panda_demo.launch.py "scene_path:=$scene" "sim_python:=$panda_python"
```

launch 默认在 `/panda` 启动仿真和 C++ 发布器。发布器完成一次发送后退出，仿真继续保持最终目标并发布状态，按 Ctrl+C 结束。可加 `send_demo:=false` 只启动仿真。启动时可设置 `steps_per_message`（默认 10）和 `heartbeat_timeout_s`（默认 0.25 s）；两个值必须为正数，心跳期限须有限。模型步长为 2 ms，默认每个定时周期推进十步，名义状态周期为 20 ms。

## 三组固定轨迹评估

使用同一固定版本模型和默认参数。每组运行 3 次，共 9 次，无随机采样。每次启动新进程，通过 `/panda/reset` 恢复 `home` 并观察时间回退，等待仿真从 reset 后推进至少 1 s，再由 C++ 发出固定命令。用独立 MCAP 记录轨迹、心跳、状态和时钟。所有轨迹点的七轴值见 [发布器](ros2_ws/src/panda_trajectory_cmd/publish_trajectory.cpp) 和运行后的 `summary.json`。

| 输入 | 运动时长 | 覆盖 |
| --- | ---: | --- |
| `single_axis` | 0.4 s | `joint1` 从 0 到 0.1 rad，其余保持 home |
| `multi_axis` | 0.8 s | 七轴同时由 home 到一个终点 |
| `multi_waypoint` | 1.2 s | 七轴由 home 经两个中间点到终点，部分关节中途反向 |

以接收命令日志中的仿真时刻为零点，在其后每 20 ms 核对状态与 `/clock`。用包里的位置点分段插值得到参考，误差为实测减参考。运动段报告每轴 RMSE 和最大绝对误差。结束后继续记录 400 ms，以最后 200 ms 的最大绝对误差作为稳态验收，阈值预先固定为每轴 0.05 rad。重复差异取同一相对时间、同一关节在三次运行中的最大两两差值。频率和 P50/P95/P99 间隔从 MCAP 接收墙钟计算，反映录包端收到状态的情况。

```bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=113
out="$cache/ros2-results/$(date +%Y%m%d-%H%M%S)-three-trajectories"
"$panda_python" "$project/ros2_smoke/evaluate.py" run "$scene" "$out"
# 从既有 MCAP 和接受日志重新生成数据表
"$panda_python" "$project/ros2_smoke/evaluate.py" analyze "$out"
# 使用已安装 Matplotlib 的 Python 绘图
python "$project/ros2_smoke/plot_tracking.py" "$out"
```

输出包括 `configuration.json`、`summary.json`、63 行七轴指标的 `joint_metrics.csv`，以及各次运行的 MCAP、日志和 `tracking.csv`。绘图脚本为每组生成一张 `tracking.png`。

2026-11-07 在本机 WSL 2、ROS2 Lyrical、MuJoCo 3.17.0、NumPy 2.5.4 上完成九次。最大稳态绝对误差为 0.01137 rad，运动段最差单轴 RMSE 为 0.04432 rad，最大瞬时误差为 0.07341 rad。状态接收频率为 49.98–50.04 Hz，各次 P99 接收间隔最高为 21.19 ms。三组重复运行的最大位置差为 `8.07e-9 rad`。本轮实测文件在 `C:/Users/PC/Documents/ChatGPT/EIP/panda-nov08-results/`。
