"""Record three fixed Panda trajectories and analyze repeatability from MCAP."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import mujoco
import numpy as np
import rclpy

from run import NAMES, NS, nanoseconds
from run_formal import (
    SAMPLE_NS, Probe, aligned_window, read_acceptance, read_bag, record, simulator_command,
    write_live_qos,
)


CASES = {"single_axis": 400_000_000, "multi_axis": 800_000_000, "multi_waypoint": 1_200_000_000}
REPEATS = 3
SETTLING_NS = 400_000_000
STEADY_WINDOW_NS = 200_000_000
STEADY_MAX_ABS_RAD = 0.05


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def analyze_run(directory, duration_ns):
    states, clocks, command, heartbeats, received_ns = read_bag(directory / "bag")
    start_ns = read_acceptance(directory / "sim_record.log")
    end_ns = duration_ns + SETTLING_NS
    samples = aligned_window(states, clocks, start_ns, end_ns)
    offsets = np.arange(SAMPLE_NS, end_ns + 1, SAMPLE_NS)
    command_times = [nanoseconds(point.time_from_start) for point in command.points]
    if command_times[-1] != duration_ns or set(command.joint_names) != set(NAMES):
        raise RuntimeError(f"Recorded command does not match the evaluation case: {directory}")
    order = [command.joint_names.index(name) for name in NAMES]
    waypoints = np.array([point.positions for point in command.points])[:, order]
    reference = np.column_stack([np.interp(offsets, command_times, axis) for axis in waypoints.T])
    actual = samples[:, 0, :]
    error = actual - reference
    active = error[offsets <= duration_ns]
    steady = error[offsets > end_ns - STEADY_WINDOW_NS]
    rmse = np.sqrt(np.mean(active ** 2, axis=0))
    peak = np.max(np.abs(active), axis=0)
    steady_max = np.max(np.abs(steady), axis=0)
    joint_rows = list(zip(NAMES, rmse, peak, np.mean(steady, axis=0), steady_max, strict=True))
    write_csv(directory / "tracking.csv",
              ("relative_s", "joint", "reference_rad", "actual_rad", "error_rad"),
              ((offset / NS, name, reference[index, axis], actual[index, axis], error[index, axis])
               for index, offset in enumerate(offsets) for axis, name in enumerate(NAMES)))

    by_stamp = {state[0]: receive for state, receive in zip(states, received_ns, strict=True)}
    wall_ns = np.array([by_stamp[start_ns + int(offset)] for offset in offsets])
    intervals_ns = np.diff(wall_ns)
    if np.any(intervals_ns <= 0):
        raise RuntimeError(f"Recorder wall timestamps are not increasing: {directory}")
    p50, p95, p99 = np.percentile(intervals_ns / 1e6, (50, 95, 99))
    report = {
        "acceptance_sim_ns": start_ns,
        "sample_count": len(samples),
        "heartbeat_count": heartbeats,
        "waypoint_times_ns": command_times,
        "waypoint_positions_rad": waypoints.tolist(),
        "state_receive_hz": float(len(intervals_ns) * NS / (wall_ns[-1] - wall_ns[0])),
        "state_interval_p50_ms": float(p50),
        "state_interval_p95_ms": float(p95),
        "state_interval_p99_ms": float(p99),
        "worst_steady_abs_rad": float(np.max(steady_max)),
    }
    return samples, joint_rows, report


def analyze(output):
    runs = []
    metrics = []
    repeatability = {}
    for case, duration_ns in CASES.items():
        case_samples = []
        for repeat in range(1, REPEATS + 1):
            samples, joint_rows, report = analyze_run(output / case / f"run-{repeat}", duration_ns)
            case_samples.append(samples)
            metrics.extend((case, repeat, *row) for row in joint_rows)
            runs.append({"case": case, "repeat": repeat, **report})
        # At each joint and aligned timestamp, max - min is the largest pairwise difference.
        spread = np.ptp(case_samples, axis=0)
        repeatability[case] = {
            "max_abs_position_delta_rad": float(np.max(spread[:, 0, :])),
            "max_abs_velocity_delta_rad_s": float(np.max(spread[:, 1, :])),
        }
    passed = all(run["worst_steady_abs_rad"] <= STEADY_MAX_ABS_RAD for run in runs)
    summary = {
        "result": "PASS" if passed else "FAIL",
        "steady_max_abs_limit_rad": STEADY_MAX_ABS_RAD,
        "run_count": len(runs),
        "worst_active_rmse_rad": float(max(row[3] for row in metrics)),
        "worst_active_peak_abs_rad": float(max(row[4] for row in metrics)),
        "worst_steady_abs_rad": max(run["worst_steady_abs_rad"] for run in runs),
        "state_receive_hz_range": [min(run["state_receive_hz"] for run in runs),
                                   max(run["state_receive_hz"] for run in runs)],
        "repeatability": repeatability,
        "runs": runs,
    }
    write_csv(output / "joint_metrics.csv",
              ("case", "repeat", "joint", "active_rmse_rad", "active_peak_abs_rad",
               "steady_mean_signed_rad", "steady_max_abs_rad"), metrics)
    write_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))
    return 0 if passed else 1


def capture(scene, output):
    model = mujoco.MjModel.from_xml_path(str(scene))
    if not math.isclose(10 * model.opt.timestep, SAMPLE_NS / NS):
        raise ValueError("The evaluation expects a 2 ms model step and ten steps per state")
    output.mkdir(parents=True)
    write_json(output / "configuration.json", {
        "scene": str(scene),
        "scene_xml_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "processor": platform.machine(),
        "ros_distro": os.environ["ROS_DISTRO"],
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "physics_step_s": model.opt.timestep,
        "state_period_s": SAMPLE_NS / NS,
        "heartbeat_timeout_s": 0.25,
        "settling_window_s": SETTLING_NS / NS,
        "steady_window_s": STEADY_WINDOW_NS / NS,
        "case_duration_ns": CASES,
        "repeats": REPEATS,
        "seed": None,
    })
    command = simulator_command(scene)
    qos = write_live_qos(output)
    with rclpy.init(args=[]):
        for case, duration_ns in CASES.items():
            for repeat in range(1, REPEATS + 1):
                directory = output / case / f"run-{repeat}"
                directory.mkdir(parents=True)
                probe = Probe(live=True)
                try:
                    print(f"Recording {case} {repeat}/{REPEATS}", flush=True)
                    record(directory, command, probe, qos, trajectory_case=case,
                           end_ns=duration_ns + SETTLING_NS)
                finally:
                    probe.destroy_node()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run_parser = modes.add_parser("run", help="Capture nine short runs and analyze the bags")
    run_parser.add_argument("scene", type=Path)
    run_parser.add_argument("output", type=Path)
    analyze_parser = modes.add_parser("analyze", help="Regenerate CSV and metrics from existing bags")
    analyze_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "run":
        capture(args.scene.resolve(strict=True), output)
    raise SystemExit(analyze(output))
