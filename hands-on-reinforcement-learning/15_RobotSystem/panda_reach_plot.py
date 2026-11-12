"""Plot nominal and delayed joint trajectories for a fixed evaluation case."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_case(path, seed):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [row for row in rows if row["seed"] == seed]
    if not rows:
        raise ValueError(f"No episode with seed {seed} in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nominal", type=Path)
    parser.add_argument("delayed", type=Path)
    parser.add_argument("output", type=Path, help="New PNG file")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    nominal = read_case(args.nominal, args.seed)
    delayed = read_case(args.delayed, args.seed)
    if nominal[0]["observation"] != delayed[0]["observation"]:
        raise ValueError("Evaluation cases have different initial observations")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col", layout="constrained")
    for column, (label, rows) in enumerate((("Nominal", nominal), ("Delayed", delayed))):
        positions, speeds = axes[:, column]
        time = [0.0, *(row["sim_time_s"] for row in rows)]
        observations = [rows[0]["observation"], *(row["next_observation"] for row in rows)]
        for joint, color in enumerate(("tab:blue", "tab:orange")):
            positions.plot(time, [obs[joint] for obs in observations], color=color, label=f"q{joint + 1}")
            positions.step(time, [obs[6 + joint] for obs in observations], where="pre",
                           color=color, alpha=.65, linestyle="--", label=f"ctrl{joint + 1}")
            positions.axhline(observations[0][4 + joint], color=color, linestyle=":", alpha=.75)
            speeds.plot(time, [abs(obs[2 + joint]) if obs[2 + joint] is not None else float("nan")
                               for obs in observations], color=color)
        speeds.axhline(.05, color="0.35", linestyle=":", label="Success threshold")
        positions.set(title=f"{label} ({rows[-1]['outcome']})", ylabel="Joint angle (rad)")
        speeds.set(xlabel="Simulation time (s)", ylabel="Absolute speed (rad/s)", ylim=(0, None))
        for axis in (positions, speeds):
            axis.grid(alpha=.2)
        positions.legend(ncol=2)
        speeds.legend()
    fig.suptitle(f"Panda position-target baseline · case {args.seed}\nDotted angle lines mark task goals")
    with args.output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
