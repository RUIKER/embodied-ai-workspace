"""Exercise one fixed Panda trajectory through ROS 2 and both bag replay modes."""

import argparse
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import mujoco as mj
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.duration import Duration
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


COMMAND = "/panda_smoke/trajectory"
STATE = "/panda_smoke/joint_states"
CLOCK = "/clock"
NAMES = [f"joint{i}" for i in range(1, 8)]
FAULTS = ["joint_names", "nonfinite", "range", "time_order"]
NS = 1_000_000_000


def nanoseconds(stamp):
    return stamp.sec * NS + stamp.nanosec


def snapshot(msg):
    if list(msg.name) != NAMES:
        raise RuntimeError(f"Unexpected JointState order: {msg.name}")
    return nanoseconds(msg.header.stamp), tuple(msg.position), tuple(msg.velocity)


def wait_until(node, ready, description, timeout=20):
    deadline = time.monotonic() + timeout
    while not ready():
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        rclpy.spin_once(node, timeout_sec=0.05)


@contextmanager
def process(log, *command):
    with log.open("wb") as output, subprocess.Popen(
        command, stdout=output, stderr=subprocess.STDOUT
    ) as child:
        try:
            yield child
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def require_success(child, log):
    if child.wait(timeout=20) != 0:
        raise RuntimeError(f"See {log}")


class Panda(Node):
    def __init__(self, scene):
        super().__init__("panda_sim_smoke")
        self.model = mj.MjModel.from_xml_path(str(scene))
        self.data = mj.MjData(self.model)
        self.home = self.model.key("home").id
        joint_ids = [self.model.joint(name).id for name in NAMES]
        self.qpos = self.model.jnt_qposadr[joint_ids]
        self.qvel = self.model.jnt_dofadr[joint_ids]
        self.actuators = [self.model.actuator(f"actuator{i}").id for i in range(1, 8)]
        for joint, actuator in zip(joint_ids, self.actuators, strict=True):
            if (self.model.actuator_trntype[actuator] != mj.mjtTrn.mjTRN_JOINT
                    or self.model.actuator_trnid[actuator, 0] != joint):
                raise RuntimeError("Unexpected Panda actuator mapping")
        self.lower, self.upper = self.model.actuator_ctrlrange[self.actuators].T
        self.clock = self.create_publisher(Clock, CLOCK, 10)
        self.states = self.create_publisher(JointState, STATE, 10)
        self.sub = self.create_subscription(JointTrajectory, COMMAND, self.receive, 10)
        self.trajectory = None
        self.accepted = 0
        self.rejected = []

    def receive(self, message):
        names = list(message.joint_names)
        if len(names) != len(NAMES) or set(names) != set(NAMES):
            self.rejected.append("joint_names")
            return
        if (nanoseconds(message.header.stamp) or len(message.points) < 2
                or any(len(point.positions) != len(NAMES) or point.velocities
                       or point.accelerations or point.effort for point in message.points)):
            self.rejected.append("format")
            return
        times = np.array([nanoseconds(point.time_from_start) for point in message.points])
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            self.rejected.append("time_order")
            return
        values = np.array([point.positions for point in message.points])[
            :, [names.index(name) for name in NAMES]
        ]
        if not np.isfinite(values).all():
            self.rejected.append("nonfinite")
            return
        if np.any(values < self.lower) or np.any(values > self.upper):
            self.rejected.append("range")
            return
        self.trajectory = times, values
        self.accepted += 1

    def wait_for_trajectory(self):
        wait_until(
            self, lambda: self.accepted and len(self.rejected) >= len(FAULTS),
            "The simulator did not receive all trajectory cases",
        )
        if self.accepted != 1 or self.rejected != FAULTS:
            raise RuntimeError(f"Unexpected command results: {self.accepted}, {self.rejected}")

    def execute(self):
        mj.mj_resetDataKeyframe(self.model, self.data, self.home)
        mj.mj_forward(self.model, self.data)
        dt_ns = round(self.model.opt.timestep * NS)
        if dt_ns != 2_000_000:
            raise RuntimeError(f"Unexpected model timestep: {dt_ns} ns")

        times, values = self.trajectory
        samples, references = [], []
        for step in range(1, 401):
            elapsed = step * dt_ns
            self.data.ctrl[self.actuators] = [
                np.interp(elapsed, times, joint) for joint in values.T
            ]
            mj.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError(f"Non-finite Panda state at step {step}")
            if step % 5 == 0:
                sec, nanosec = divmod(elapsed, NS)
                stamp = Time(sec=sec, nanosec=nanosec)
                state = JointState(
                    name=NAMES, position=self.data.qpos[self.qpos].tolist(),
                    velocity=self.data.qvel[self.qvel].tolist(),
                )
                state.header.stamp = stamp
                self.clock.publish(Clock(clock=stamp))
                self.states.publish(state)
                samples.append(snapshot(state))
                references.append(self.data.ctrl[self.actuators].copy())
                time.sleep(0.01)

        if any(warning.number for warning in self.data.warning):
            raise RuntimeError("MuJoCo reported a warning")
        errors = np.array([positions for _, positions, _ in samples]) - references
        steady = np.mean(np.abs(errors[-20:]), axis=0)
        if np.max(steady) > 0.05 or samples[-1][1][0] <= 0.05:
            raise RuntimeError("The fixed trajectory missed its smoke threshold")
        return samples, {
            "rmse_rad": dict(zip(NAMES, np.sqrt(np.mean(errors**2, axis=0)).tolist())),
            "peak_abs_error_rad": dict(zip(NAMES, np.max(np.abs(errors), axis=0).tolist())),
            "hold_mean_abs_error_rad": dict(zip(NAMES, steady.tolist())),
        }


def record_run(simulator, output):
    log = output / "record.log"
    with process(
        log, "ros2", "bag", "record", "-s", "mcap", "-o", str(output / "bag"),
        "--topics", COMMAND, STATE, CLOCK,
    ) as recorder:
        wait_until(
            simulator, lambda: all(simulator.count_subscribers(t) for t in (STATE, CLOCK)),
            "rosbag2 did not subscribe to state and clock",
        )
        publisher_log = output / "publisher.log"
        with process(
            publisher_log, "ros2", "run", "panda_trajectory_smoke", "publish_trajectory"
        ) as publisher:
            simulator.wait_for_trajectory()
            require_success(publisher, publisher_log)
        samples, metrics = simulator.execute()
        if not all(pub.wait_for_all_acked(Duration(seconds=3))
                   for pub in (simulator.states, simulator.clock)):
            raise RuntimeError("State or clock was not acknowledged by the bag recorder")
        recorder.send_signal(signal.SIGINT)
        require_success(recorder, log)
    return samples, metrics


def play(output, log, *topics):
    return process(
        log, "ros2", "bag", "play", str(output / "bag"), "--delay", "1",
        "--topics", *topics,
    )


def check_state_replay(output, expected):
    probe = Node("panda_state_replay_probe")
    states, clocks = [], []
    try:
        probe.create_subscription(JointState, STATE, lambda msg: states.append(snapshot(msg)), 10)
        probe.create_subscription(Clock, CLOCK, lambda msg: clocks.append(nanoseconds(msg.clock)), 10)
        log = output / "play_states.log"
        with play(output, log, STATE, CLOCK) as player:
            wait_until(
                probe, lambda: len(states) >= len(expected) and len(clocks) >= len(expected),
                "Recorded state or clock samples were missing on playback",
            )
            require_success(player, log)
        if states != expected or clocks != [stamp for stamp, _, _ in expected]:
            raise RuntimeError("Recorded state or clock samples differed on playback")
    finally:
        probe.destroy_node()


def run(scene, output):
    simulator = Panda(scene)
    try:
        expected, metrics = record_run(simulator, output)
    finally:
        simulator.destroy_node()

    with (output / "bag_info.txt").open("w", encoding="utf-8") as info:
        subprocess.run(
            ["ros2", "bag", "info", str(output / "bag")], check=True, stdout=info, timeout=15,
        )
    check_state_replay(output, expected)

    simulator = Panda(scene)
    try:
        log = output / "play_commands.log"
        with play(output, log, COMMAND) as player:
            simulator.wait_for_trajectory()
            require_success(player, log)
        repeated, _ = simulator.execute()
    finally:
        simulator.destroy_node()
    np.testing.assert_array_equal([stamp for stamp, _, _ in repeated],
                                  [stamp for stamp, _, _ in expected])
    state_delta = np.array([sample[1:] for sample in repeated]) - np.array(
        [sample[1:] for sample in expected]
    )
    np.testing.assert_allclose(state_delta, 0, atol=1e-12, rtol=0)
    return {
        "result": "PASS",
        "ros_distro": os.environ["ROS_DISTRO"],
        "mujoco": mj.__version__,
        "numpy": np.__version__,
        "accepted_trajectories": 1,
        "rejected_cases": FAULTS,
        "state_replay_samples": len(expected),
        "command_reexecution_samples": len(repeated),
        "reexecution_max_abs_position_delta_rad": float(np.max(np.abs(state_delta[:, 0]))),
        "reexecution_max_abs_velocity_delta_rad_s": float(np.max(np.abs(state_delta[:, 1]))),
        "simulation_seconds": expected[-1][0] / NS,
        **metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Path to the pinned Panda scene.xml")
    parser.add_argument("output", type=Path, help="New directory for the bag and test evidence")
    options = parser.parse_args()
    options.output.mkdir(parents=True)
    rclpy.init(args=[])
    try:
        report = json.dumps(run(options.scene, options.output), indent=2)
        (options.output / "summary.json").write_text(report + "\n", encoding="utf-8")
        print(report)
    finally:
        rclpy.shutdown()
