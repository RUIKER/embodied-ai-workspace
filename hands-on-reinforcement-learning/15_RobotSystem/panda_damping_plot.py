"""Plot recorded Panda damping experiments."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_run(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    names = ("baseline_1", "baseline_2", "damped_1", "damped_2")
    rows = {name: [] for name in names}
    with (directory / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        has_torque = "joint_actuator_torque_Nm" in reader.fieldnames
        for row in reader:
            if row["condition"] not in rows:
                raise ValueError(f"Unexpected condition {row['condition']}")
            values = [
                float(row["sample_time_s"]), float(row["applied_target_rad"]),
                float(row["position_rad"]), float(row["velocity_rad_s"]),
            ]
            if has_torque:
                values.extend([float(row["control_time_s"]), float(row["joint_actuator_torque_Nm"])])
            rows[row["condition"]].append(values)
    series = {name: np.asarray(values, dtype=float) for name, values in rows.items()}
    reference = series["baseline_1"]
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError("Expected at least two baseline samples")
    for name, data in series.items():
        if data.shape != reference.shape or not np.isfinite(data).all():
            raise ValueError(f"Invalid samples for {name}")
        if not np.all(np.diff(data[:, 0]) > 0):
            raise ValueError(f"Sample times do not increase for {name}")
        np.testing.assert_allclose(data[:, :2], reference[:, :2], atol=1e-10, rtol=0)
    return summary, series


def plot_run(directory, output):
    summary, series = read_run(directory)
    base, repeat = series["baseline_1"], series["baseline_2"]
    damped, damped_repeat = series["damped_1"], series["damped_2"]
    time = base[:, 0]
    b0, b1 = summary["baseline_damping"], summary["changed_damping"]
    has_torque = base.shape[1] == 6
    figure, axes = plt.subplots(4 if has_torque else 3, 2, figsize=(12, 13 if has_torque else 10), sharex=True, layout="constrained")

    axes[0, 0].plot(time, base[:, 1], "--", color="black", label="Applied target")
    axes[0, 0].plot(time, base[:, 2], color="#1768ac", label=f"Original b={b0:g}")
    axes[0, 0].plot(time, damped[:, 2], color="#d45b12", label=f"Changed b={b1:g}")
    axes[0, 0].set(title="Joint position", ylabel="Position (rad)")
    axes[0, 0].legend(loc="upper left")

    axes[0, 1].plot(time, (damped[:, 2] - base[:, 2]) * 1e6, color="#d45b12")
    axes[0, 1].set(title="Changed minus original", ylabel="Position difference (µrad)")

    axes[1, 0].plot(time, base[:, 3], color="#1768ac", label="Original")
    axes[1, 0].plot(time, damped[:, 3], color="#d45b12", label="Changed")
    axes[1, 0].set(title="Joint velocity", ylabel="Velocity (rad/s)")
    axes[1, 0].legend()

    axes[1, 1].plot(time, -b0 * base[:, 3], color="#1768ac", label="Original")
    axes[1, 1].plot(time, -b1 * damped[:, 3], color="#d45b12", label="Changed")
    axes[1, 1].set(title="Damping torque component: −b × velocity", ylabel="Torque (N·m)")
    axes[1, 1].legend()

    axes[2, 0].plot(time, base[:, 2] - base[:, 1], color="#1768ac", label="Original")
    axes[2, 0].plot(time, damped[:, 2] - damped[:, 1], color="#d45b12", label="Changed")
    axes[2, 0].set(title="Position minus applied target", ylabel="Tracking error (rad)")
    axes[2, 0].legend()

    axes[2, 1].plot(time, (repeat[:, 2] - base[:, 2]) * 1e6,
                    color="#1768ac", linewidth=2.4, label="Original repeat")
    axes[2, 1].plot(time, (damped_repeat[:, 2] - damped[:, 2]) * 1e6,
                    color="#d45b12", linestyle="--", label="Changed repeat")
    axes[2, 1].set(title="Same-setting repeat minus first run", ylabel="Position difference (µrad)")
    axes[2, 1].legend()

    if has_torque:
        axes[3, 0].plot(base[:, 4], base[:, 5], color="#1768ac", label="Original")
        axes[3, 0].plot(damped[:, 4], damped[:, 5], color="#d45b12", label="Changed")
        axes[3, 0].set(title="Joint1 actuator torque", ylabel="Torque (N·m)")
        axes[3, 0].legend()
        axes[3, 1].plot(base[:, 4], damped[:, 5] - base[:, 5], color="#d45b12")
        axes[3, 1].set(title="Actuator torque: changed minus original", ylabel="Torque difference (N·m)")

    for axis in axes.flat:
        axis.axhline(0, color="#888888", linewidth=0.65)
        axis.axvline(summary["trajectory_duration_s"], color="#999999", linestyle=":", linewidth=1)
        axis.grid(alpha=0.22)
    for axis in axes[-1]:
        axis.set_xlabel("Simulation time since command start (s)")
    figure.suptitle("Panda joint1 response to damping", fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing trajectory.csv and summary.json")
    parser.add_argument("--output", type=Path, help="PNG path (defaults to comparison.png in directory)")
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    output = args.output if args.output else directory / "comparison.png"
    plot_run(directory, output)
    print(f"Saved {output}")
