"""Record and replay one fixed trajectory using the formal Panda ROS 2 nodes."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectory

from run import nanoseconds, process, require_success, snapshot, wait_until


COMMAND = "/panda/trajectory"
HEARTBEAT = "/panda/trajectory_heartbeat"
STATE = "/panda/joint_states"
CLOCK = "/clock"
TYPES = {
    COMMAND: "trajectory_msgs/msg/JointTrajectory",
    HEARTBEAT: "std_msgs/msg/Empty",
    STATE: "sensor_msgs/msg/JointState",
    CLOCK: "rosgraph_msgs/msg/Clock",
}
SAMPLE_NS = 20_000_000
END_NS = 800_000_000
ENDPOINT_TOLERANCE_RAD = 0.005
ACCEPTANCE = re.compile(r"Accepted JointTrajectory .* at simulation stamp (\d+) ns")
COMPLETION = "Trajectory complete; holding final targets"


class Probe(Node):
    def __init__(self, live):
        super().__init__("panda_formal_bag_probe")
        self.states = []
        self.clocks = []
        clock_qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT if live else ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            JointState, STATE, lambda msg: self.states.append(snapshot(msg)), 100,
        )
        self.create_subscription(
            Clock, CLOCK, lambda msg: self.clocks.append(nanoseconds(msg.clock)), clock_qos,
        )

    def verify_publishers(self, name):
        for topic in (STATE, CLOCK):
            endpoints = self.get_publishers_info_by_topic(topic)
            if [endpoint.node_name for endpoint in endpoints] != [name]:
                raise RuntimeError(f"Expected {name} as the sole publisher on {topic}")


def stop(child, log):
    child.send_signal(signal.SIGINT)
    require_success(child, log)


def read_acceptance(log):
    messages = log.read_text(encoding="utf-8")
    matches = ACCEPTANCE.findall(messages)
    if (len(matches) != 1 or messages.count(COMPLETION) != 1
            or "timed out" in messages or "Rejected JointTrajectory" in messages):
        raise RuntimeError(f"Expected one completed trajectory with no rejection or timeout: {log}")
    return int(matches[0])


def record(output, sim_command, probe, qos):
    static_topics = output / "record_topics.yaml"
    static_topics.write_text(
        json.dumps({"static_topics_and_types_list": list(TYPES.items())}), encoding="utf-8",
    )
    sim_log = output / "sim_record.log"
    record_log = output / "record.log"
    with process(sim_log, *sim_command) as simulator:
        wait_until(probe, lambda: probe.states and probe.clocks, "Formal simulator did not start")
        probe.verify_publishers("panda_sim")
        with process(
            record_log, "ros2", "bag", "record", "-s", "mcap", "-o", str(output / "bag"),
            "--static-topics-path", str(static_topics), "--no-discovery",
            "--qos-profile-overrides-path", str(qos), "--disable-keyboard-controls",
        ) as recorder:
            wait_until(
                probe,
                lambda: all(
                    any(endpoint.node_name == "rosbag2_recorder"
                        for endpoint in probe.get_subscriptions_info_by_topic(topic))
                    for topic in TYPES
                ) and probe.states[-1][0] >= 1_000_000_000,
                "Recorder did not discover all four subscriptions",
            )
            publish_log = output / "publisher.log"
            with process(
                publish_log, "ros2", "run", "panda_trajectory_cmd", "publish_trajectory",
                "--ros-args", "-r", "__ns:=/panda",
            ) as publisher:
                require_success(publisher, publish_log)
            wait_until(probe, lambda: COMPLETION in sim_log.read_text(encoding="utf-8"),
                       f"The simulator did not complete its trajectory: {sim_log}")
            start_ns = read_acceptance(sim_log)
            wait_until(probe, lambda: probe.states[-1][0] > start_ns + END_NS,
                       "Missing recorded settling window")
            stop(recorder, record_log)
        stop(simulator, sim_log)
    return start_ns


def read_bag(bag):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    if {topic.name: topic.type for topic in reader.get_all_topics_and_types()} != TYPES:
        raise RuntimeError("Unexpected topic set in the formal MCAP")
    states, clocks, commands, heartbeats = [], [], [], []
    while reader.has_next():
        topic, payload, receive_ns, _send_ns = reader.read_next_ext()
        if topic == STATE:
            states.append(snapshot(deserialize_message(payload, JointState)))
        elif topic == CLOCK:
            clocks.append(nanoseconds(deserialize_message(payload, Clock).clock))
        elif topic == COMMAND:
            commands.append((receive_ns, deserialize_message(payload, JointTrajectory)))
        else:
            deserialize_message(payload, Empty)
            heartbeats.append(receive_ns)
    if len(commands) != 1 or len(heartbeats) < 2 or not states or not clocks:
        raise RuntimeError("MCAP is missing the fixed command, heartbeat, state or clock")
    command_ns, command = commands[0]
    if heartbeats[-1] - command_ns < nanoseconds(command.points[-1].time_from_start):
        raise RuntimeError("The recorded heartbeat does not span the command duration")
    return states, clocks, command, len(heartbeats)


def play(output, log, qos, *topics, options=()):
    return process(
        log, "ros2", "bag", "play", str(output / "bag"), "--delay", "1",
        "--disable-keyboard-controls", "--progress-bar-update-rate", "0", *options,
        "--qos-profile-overrides-path", str(qos), "--topics", *topics,
    )


def check_state_replay(output, states, clocks, qos):
    probe = Probe(live=False)
    log = output / "play_states.log"
    try:
        with play(output, log, qos, STATE, CLOCK, options=("--wait-for-all-acked", "1500")) as player:
            wait_until(probe, lambda: probe.states and probe.clocks,
                       "The player did not publish state and clock")
            probe.verify_publishers("rosbag2_player")
            wait_until(probe, lambda: len(probe.states) >= len(states) and len(probe.clocks) >= len(clocks),
                       "Recorded state or clock samples were missing on playback")
            require_success(player, log)
        if probe.states != states or probe.clocks != clocks:
            raise RuntimeError("Replayed state or clock differed from the MCAP")
    finally:
        probe.destroy_node()


def reexecute(output, sim_command, qos):
    probe = Probe(live=True)
    sim_log = output / "sim_reexecute.log"
    player_log = output / "play_commands.log"
    try:
        with process(sim_log, *sim_command) as simulator:
            wait_until(probe, lambda: probe.states and probe.clocks, "Restarted simulator did not start")
            probe.verify_publishers("panda_sim")
            with play(output, player_log, qos, COMMAND, HEARTBEAT) as player:
                wait_until(probe, lambda: probe.count_publishers(COMMAND) and probe.count_publishers(HEARTBEAT),
                           "The player did not advertise command and heartbeat")
                probe.verify_publishers("panda_sim")
                wait_until(probe, lambda: COMPLETION in sim_log.read_text(encoding="utf-8"),
                           f"The restarted simulator did not complete its trajectory: {sim_log}")
                start_ns = read_acceptance(sim_log)
                wait_until(probe, lambda: probe.states[-1][0] >= start_ns + END_NS,
                           "Missing reexecution settling window")
                require_success(player, player_log)
            stop(simulator, sim_log)
        return probe.states, probe.clocks, start_ns
    finally:
        probe.destroy_node()


def aligned_window(states, clocks, start_ns):
    window = {stamp - start_ns: (positions, velocities) for stamp, positions, velocities in states}
    stamps = range(SAMPLE_NS, END_NS + 1, SAMPLE_NS)
    if any(offset not in window for offset in stamps):
        raise RuntimeError("Missing a JointState in the command-relative comparison window")
    if not {start_ns + offset for offset in stamps}.issubset(clocks):
        raise RuntimeError("Missing a matching /clock sample in the comparison window")
    samples = np.array([window[offset] for offset in stamps])
    if not np.isfinite(samples).all():
        raise RuntimeError("Non-finite state in the command-relative comparison window")
    return samples


def run(scene, output):
    prefix = Path(subprocess.check_output(["ros2", "pkg", "prefix", "panda_sim"], text=True, timeout=10).strip())
    sim_command = [sys.executable, str(prefix / "lib/panda_sim/panda_sim_node"),
                   "--ros-args", "-r", "__ns:=/panda", "-p", f"scene_path:={scene}"]
    qos = output / "live_qos.yaml"
    qos.write_text(json.dumps({topic: {"reliability": "best_effort", "depth": 100}
                              for topic in (CLOCK, HEARTBEAT)}), encoding="utf-8")
    playback_qos = output / "state_playback_qos.yaml"
    playback_qos.write_text(json.dumps({topic: {"reliability": "reliable", "depth": 100}
                                       for topic in (CLOCK, STATE)}), encoding="utf-8")
    probe = Probe(live=True)
    try:
        start_ns = record(output, sim_command, probe, qos)
    finally:
        probe.destroy_node()

    bag = output / "bag"
    with (output / "bag_info.txt").open("w", encoding="utf-8") as info:
        subprocess.run(["ros2", "bag", "info", str(bag)], check=True, stdout=info, timeout=15)
    states, clocks, command, heartbeats = read_bag(bag)
    baseline = aligned_window(states, clocks, start_ns)
    check_state_replay(output, states, clocks, playback_qos)
    repeated, repeated_clocks, repeated_start_ns = reexecute(output, sim_command, qos)
    replay = aligned_window(repeated, repeated_clocks, repeated_start_ns)
    joint1_target = command.points[-1].positions[command.joint_names.index("joint1")]
    endpoints = [float(baseline[-1, 0, 0]), float(replay[-1, 0, 0])]
    np.testing.assert_allclose(endpoints, joint1_target, atol=ENDPOINT_TOLERANCE_RAD, rtol=0)
    delta = np.max(np.abs(replay - baseline), axis=(0, 2))
    return {
        "result": "PASS",
        "ros_distro": os.environ["ROS_DISTRO"],
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "command_messages": 1,
        "heartbeat_messages": heartbeats,
        "state_replay_samples": len(states),
        "clock_replay_samples": len(clocks),
        "command_reexecution_samples": len(replay),
        "recorded_acceptance_sim_ns": start_ns,
        "reexecution_acceptance_sim_ns": repeated_start_ns,
        "recorded_joint1_endpoint_rad": endpoints[0],
        "reexecuted_joint1_endpoint_rad": endpoints[1],
        "reexecution_max_abs_position_delta_rad": float(delta[0]),
        "reexecution_max_abs_velocity_delta_rad_s": float(delta[1]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Path to the pinned Panda scene.xml")
    parser.add_argument("output", type=Path, help="New directory for the bag and test evidence")
    options = parser.parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True)
    with rclpy.init(args=[]):
        report = json.dumps(run(options.scene.resolve(strict=True), output), indent=2)
        (output / "summary.json").write_text(report + "\n", encoding="utf-8")
        print(report)
