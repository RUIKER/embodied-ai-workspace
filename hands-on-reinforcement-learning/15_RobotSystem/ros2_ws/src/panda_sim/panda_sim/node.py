"""Execute position trajectories and publish live Panda joint states."""

import time

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock as ClockMessage
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory


STEPS_PER_MESSAGE = 10
NANOSECONDS_PER_SECOND = 1_000_000_000
HEARTBEAT_TIMEOUT = 0.25


def trajectory_points(message, names, lower, upper):
    incoming_names = list(message.joint_names)
    if len(incoming_names) != len(names) or set(incoming_names) != set(names):
        raise ValueError("joint_names")
    if message.header.stamp.sec or message.header.stamp.nanosec:
        raise ValueError("nonzero_stamp")
    if len(message.points) < 2 or any(
        len(point.positions) != len(names)
        or point.velocities or point.accelerations or point.effort
        or point.time_from_start.nanosec >= NANOSECONDS_PER_SECOND
        for point in message.points
    ):
        raise ValueError("format")

    times = np.array([
        point.time_from_start.sec * NANOSECONDS_PER_SECOND + point.time_from_start.nanosec
        for point in message.points
    ])
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("time_order")

    order = [incoming_names.index(name) for name in names]
    positions = np.array([point.positions for point in message.points])[:, order]
    if not np.isfinite(positions).all():
        raise ValueError("nonfinite")
    if np.any(positions < lower) or np.any(positions > upper):
        raise ValueError("range")
    return times, positions


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = Node("panda_sim")
            scene_path = node.declare_parameter("scene_path", "").value
            if not scene_path:
                raise ValueError("Set the scene_path parameter to the Panda scene.xml")

            model = mujoco.MjModel.from_xml_path(scene_path)
            data = mujoco.MjData(model)
            home = model.key("home").id
            mujoco.mj_resetDataKeyframe(model, data, home)
            mujoco.mj_forward(model, data)

            names = [f"joint{i}" for i in range(1, 8)]
            joint_ids = [model.joint(name).id for name in names]
            position_indices = model.jnt_qposadr[joint_ids]
            velocity_indices = model.jnt_dofadr[joint_ids]
            actuators = [model.actuator(f"actuator{i}").id for i in range(1, 8)]
            for joint_id, actuator_id in zip(joint_ids, actuators, strict=True):
                if (model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
                        or model.actuator_trnid[actuator_id, 0] != joint_id):
                    raise ValueError("Unexpected Panda actuator mapping")
            lower = np.maximum(model.jnt_range[joint_ids, 0], model.actuator_ctrlrange[actuators, 0])
            upper = np.minimum(model.jnt_range[joint_ids, 1], model.actuator_ctrlrange[actuators, 1])
            publisher = node.create_publisher(JointState, "joint_states", 10)
            latest_only = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            clock_publisher = node.create_publisher(ClockMessage, "/clock", latest_only)
            active = None
            heartbeat_deadline = 0.0

            def publish_state():
                nonlocal active
                if active is not None and time.monotonic() >= heartbeat_deadline:
                    active = None
                    node.get_logger().warning("Trajectory heartbeat timed out; holding last targets")

                for _ in range(STEPS_PER_MESSAGE):
                    if active is not None:
                        start_time, times, positions = active
                        elapsed = round((data.time - start_time) * NANOSECONDS_PER_SECOND)
                        data.ctrl[actuators] = [
                            np.interp(elapsed, times, axis) for axis in positions.T
                        ]
                        if elapsed >= times[-1]:
                            active = None
                            node.get_logger().info("Trajectory complete; holding final targets")
                    mujoco.mj_step(model, data)

                sec, nanosec = divmod(
                    round(data.time * NANOSECONDS_PER_SECOND), NANOSECONDS_PER_SECOND
                )
                stamp = Time(sec=sec, nanosec=nanosec)
                clock_publisher.publish(ClockMessage(clock=stamp))
                message = JointState()
                message.header.stamp = stamp
                message.name = names
                message.position = data.qpos[position_indices].tolist()
                message.velocity = data.qvel[velocity_indices].tolist()
                publisher.publish(message)

            def receive_trajectory(message):
                nonlocal active, heartbeat_deadline
                if active is not None:
                    node.get_logger().warning("Rejected JointTrajectory: busy")
                    return
                try:
                    times, positions = trajectory_points(message, names, lower, upper)
                except ValueError as exc:
                    node.get_logger().warning(f"Rejected JointTrajectory: {exc}")
                    return
                if not np.allclose(positions[0], data.ctrl[actuators], rtol=0, atol=1e-6):
                    node.get_logger().warning("Rejected JointTrajectory: start_position")
                    return

                active = (data.time, times, positions)
                heartbeat_deadline = time.monotonic() + HEARTBEAT_TIMEOUT
                joint_names = ", ".join(names)
                node.get_logger().info(
                    f"Accepted JointTrajectory with {len(times)} points "
                    f"for [{joint_names}] at simulation stamp "
                    f"{round(data.time * NANOSECONDS_PER_SECOND)} ns"
                )

            def receive_heartbeat(_message):
                nonlocal heartbeat_deadline
                now = time.monotonic()
                if active is not None and now < heartbeat_deadline:
                    heartbeat_deadline = now + HEARTBEAT_TIMEOUT

            def reset(_request, response):
                nonlocal trajectory_subscription, heartbeat_deadline
                if active is not None:
                    response.message = "Trajectory active"
                    return response

                # A new volatile reader discards commands queued before the reset.
                node.destroy_subscription(trajectory_subscription)
                mujoco.mj_resetDataKeyframe(model, data, home)
                mujoco.mj_forward(model, data)
                heartbeat_deadline = 0.0
                trajectory_subscription = node.create_subscription(
                    JointTrajectory, "trajectory", receive_trajectory, 10
                )
                response.success = True
                response.message = "Reset to home"
                node.get_logger().info(response.message)
                return response

            trajectory_subscription = node.create_subscription(
                JointTrajectory, "trajectory", receive_trajectory, 10
            )
            node.create_subscription(Empty, "trajectory_heartbeat", receive_heartbeat, latest_only)
            node.create_service(Trigger, "reset", reset)
            period = STEPS_PER_MESSAGE * model.opt.timestep
            timer_clock = Clock(clock_type=ClockType.STEADY_TIME)
            node.create_timer(period, publish_state, clock=timer_clock)
            node.get_logger().info(f"Publishing Panda joint states every {period:.3f} s")
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
