"""Start the Panda simulator and optionally send one demo trajectory."""

from shlex import quote

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_nodes(context):
    namespace = LaunchConfiguration("namespace")
    simulator = Node(
        package="panda_sim",
        executable="panda_sim_node",
        prefix=quote(LaunchConfiguration("sim_python").perform(context)),
        namespace=namespace,
        output="screen",
        parameters=[{
            "scene_path": ParameterValue(LaunchConfiguration("scene_path"), value_type=str),
            "steps_per_message": ParameterValue(LaunchConfiguration("steps_per_message"), value_type=int),
            "heartbeat_timeout_s": ParameterValue(
                LaunchConfiguration("heartbeat_timeout_s"), value_type=float,
            ),
        }],
    )
    publisher = Node(
        package="panda_trajectory_cmd",
        executable="publish_trajectory",
        namespace=namespace,
        output="screen",
        condition=IfCondition(LaunchConfiguration("send_demo")),
    )

    def publisher_exited(event, _context):
        if event.returncode != 0:
            return [Shutdown(reason=f"Trajectory publisher exited with code {event.returncode}")]
        return []

    return [
        RegisterEventHandler(OnProcessExit(
            target_action=simulator, on_exit=[Shutdown(reason="Panda simulator exited")],
        )),
        RegisterEventHandler(OnProcessExit(target_action=publisher, on_exit=publisher_exited)),
        simulator,
        publisher,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scene_path", description="Path to the pinned Panda scene.xml"),
        DeclareLaunchArgument("sim_python", description="Python interpreter with MuJoCo and rclpy"),
        DeclareLaunchArgument("namespace", default_value="panda"),
        DeclareLaunchArgument("steps_per_message", default_value="10"),
        DeclareLaunchArgument("heartbeat_timeout_s", default_value="0.25"),
        DeclareLaunchArgument("send_demo", default_value="true"),
        OpaqueFunction(function=launch_nodes),
    ])
