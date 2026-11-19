"""Export the deterministic SAC actor for ROS 2 environments without PyTorch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from panda_reach import model_info, write_json


ACTOR_KEYS = ("fc1.weight", "fc1.bias", "fc_mu.weight", "fc_mu.bias",
              "action_scale", "action_bias")


def model_digest(scene):
    hashes = json.dumps(model_info(scene)["xml_sha256"], sort_keys=True)
    return sha256(hashes.encode()).hexdigest()


def actor_action(weights, observation):
    state = np.asarray(observation, dtype=np.float32)
    hidden = np.maximum(weights["fc1.weight"] @ state + weights["fc1.bias"], 0)
    mean = weights["fc_mu.weight"] @ hidden + weights["fc_mu.bias"]
    return weights["action_bias"] + weights["action_scale"] * np.tanh(mean)


def load_actor(path, scene):
    with np.load(path, allow_pickle=False) as archive:
        if archive["model_digest"].item() != model_digest(scene):
            raise ValueError("Actor export uses a different Panda model")
        weights = {key: archive[key] for key in ACTOR_KEYS}
        checkpoint_sha256 = archive["checkpoint_sha256"].item()
    return weights, checkpoint_sha256


def export_actor(scene, checkpoint, reference_log, case_ids, output):
    import torch

    from panda_reach import deterministic_action
    from panda_sac_smoke import reload_agent

    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    info = model_info(scene)
    if any(saved["run_config"].get(key) != info[key]
           for key in ("xml_sha256", "action_scale_rad", "max_episode_steps")):
        raise ValueError("Checkpoint uses a different Panda model or task protocol")
    checkpoint_sha256 = sha256(checkpoint.read_bytes()).hexdigest()
    agent = reload_agent(checkpoint)
    state = agent.actor.state_dict()
    weights = {key: state[key].numpy() for key in ACTOR_KEYS}
    if not (np.array_equal(weights["action_scale"], [1, 1])
            and np.array_equal(weights["action_bias"], [0, 0])):
        raise ValueError("Expected two normalized actor outputs")

    with reference_log.open(encoding="utf-8") as stream:
        rows = [row for line in stream if (row := json.loads(line))["seed"] in case_ids]
    if {row["seed"] for row in rows} != set(case_ids):
        raise ValueError("Each requested case must occur in the reference log")
    max_delta = 0.0
    for row in rows:
        observation = row["observation"]
        reference = deterministic_action(agent, observation)
        np.testing.assert_allclose(reference, row["action_normalized"], atol=1e-7, rtol=0)
        delta = np.max(np.abs(reference - actor_action(weights, observation)))
        max_delta = max(max_delta, float(delta))
    if max_delta > 1e-6:
        raise RuntimeError(f"NumPy actor differs from the PyTorch actor by {max_delta:g}")

    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "actor.npz", **weights, model_digest=model_digest(scene),
             checkpoint_sha256=checkpoint_sha256)
    report = {"checkpoint_sha256": checkpoint_sha256, "model_digest": model_digest(scene),
              "case_ids": case_ids, "reference_observations": len(rows),
              "max_abs_action_difference": max_delta, "tolerance": 1e-6}
    write_json(output / "export.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference_log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", dest="case_ids", type=int, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(export_actor(args.scene, args.checkpoint, args.reference_log,
                                  args.case_ids, args.output), indent=2))


if __name__ == "__main__":
    main()
