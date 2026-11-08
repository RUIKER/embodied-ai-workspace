"""Plot the reference and measured joint angles exported by evaluate.py."""

import argparse
import csv
import json
from pathlib import Path

from matplotlib import pyplot as plt


def plot(output):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for case in summary["repeatability"]:
        runs = [run for run in summary["runs"] if run["case"] == case]
        figure, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True, layout="constrained")
        for run in runs:
            tracking = output / case / f'run-{run["repeat"]}' / "tracking.csv"
            with tracking.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for index, axis in enumerate(axes.flat[:7], start=1):
                joint = f"joint{index}"
                samples = [row for row in rows if row["joint"] == joint]
                seconds = [float(row["relative_s"]) for row in samples]
                actual = [float(row["actual_rad"]) for row in samples]
                axis.plot(seconds, actual, alpha=0.75, label=f'Run {run["repeat"]}')
                if run is runs[-1]:
                    axis.plot(seconds, [float(row["reference_rad"]) for row in samples],
                              "k--", linewidth=1.1, label="Reference")
                    axis.axvline(run["waypoint_times_ns"][-1] / 1e9, color="0.7", linewidth=0.8)
                    axis.set_title(joint)
                    axis.set_ylabel("Position (rad)")
                    axis.grid(alpha=0.2)
        legend = axes.flat[-1]
        legend.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        legend.legend(handles, labels, loc="center", frameon=False, ncol=2)
        figure.supxlabel("Simulation time since trajectory acceptance (s)")
        figure.suptitle(f"Panda tracking — {case.replace('_', ' ')}")
        figure.savefig(output / case / "tracking.png", dpi=140)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Evaluation directory containing summary.json")
    plot(parser.parse_args().output)
