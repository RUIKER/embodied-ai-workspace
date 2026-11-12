"""Train Panda two-joint SAC or compare controllers on fixed reaching cases."""

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter

import mujoco
import numpy as np

from panda_two_joint_env import PandaTwoJointEnv


ACTION_SCALE = 0.02
MAX_STEPS = 250
TRAIN_GOALS = [(x, y) for x in (-0.16, -0.08, 0.08, 0.16) for y in (-0.16, -0.08, 0.08, 0.16)]
EVAL_CASES = {
    "validation": [(101, (0.10, 0.10)), (102, (-0.10, 0.10))],
    "test": [(201, (0.12, 0.14)), (202, (-0.14, 0.12)),
             (203, (0.14, -0.12)), (204, (-0.12, -0.14))],
}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")


def vector(value):
    return [float(x) if np.isfinite(x) else None for x in value]


def model_info(scene):
    xml = {"scene": sha256(scene.read_bytes()).hexdigest()}
    panda_xml = scene.with_name("panda.xml")
    if panda_xml.is_file():
        xml["panda"] = sha256(panda_xml.read_bytes()).hexdigest()
    return {"xml_sha256": xml, "mujoco": mujoco.__version__, "numpy": np.__version__,
            "step_s": 0.02, "action_scale_rad": ACTION_SCALE, "max_episode_steps": MAX_STEPS}


def deterministic_action(agent, observation):
    import torch

    with torch.inference_mode():
        state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return agent.actor.deterministic_action(state).squeeze(0).numpy()


def train(args):
    import torch
    from panda_sac_smoke import CONFIG, NETWORKS, ReplayBuffer, SAC, checkpoint, reload_agent

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    goal_rng = np.random.default_rng(args.seed)
    action_rng = np.random.default_rng(args.seed + 1)
    warmup, batch, capacity = 256, 64, 100_000
    if args.steps <= warmup:
        raise ValueError(f"--steps must exceed the {warmup}-step warmup")
    info = model_info(args.scene)
    env = PandaTwoJointEnv(args.scene, max_steps=MAX_STEPS)
    agent, replay = SAC(**CONFIG), ReplayBuffer(capacity)
    initial_actor = [p.detach().clone() for p in agent.actor.parameters()]
    config = {**info, "steps": args.steps, "seed": args.seed, "action_rng_seed": args.seed + 1,
              "warmup": warmup, "batch": batch, "replay_capacity": capacity,
              "goals_rad": TRAIN_GOALS, "torch": str(torch.__version__), "sac": CONFIG}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", config)

    def reset():
        goal = TRAIN_GOALS[goal_rng.integers(len(TRAIN_GOALS))]
        return env.reset(goal=goal)[0]

    observation = reset()
    reference = observation.copy()
    episode_return = updates = 0
    with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as log:
        start = perf_counter()
        for step in range(1, args.steps + 1):
            if step == warmup + 1:
                update_start = perf_counter()
            action = action_rng.uniform(-1, 1, 2) if step <= warmup else agent.take_action(observation)
            next_obs, reward, terminated, truncated, result = env.step(ACTION_SCALE * action)
            episode_return += reward
            if terminated or truncated or step == args.steps:
                json_line(log, {"step": step, "goal_rad": vector(env.goal), "return": episode_return,
                                "episode_steps": env.steps, "terminated": terminated, "truncated": truncated,
                                "outcome": result["outcome"] if terminated or truncated else "partial",
                                "physics_warning": result["physics_warning"],
                                "actual_limit_violation": result["actual_limit_violation"]})
            if not np.isfinite(next_obs).all() or result["outcome"] == "failure":
                raise RuntimeError(f"Aborting training on simulation failure at step {step}")
            replay.add(observation, action, reward, next_obs, terminated)
            if step > warmup:
                agent.update(replay.sample(batch))
                updates += 1
            observation = next_obs
            if terminated or truncated:
                observation = reset()
                episode_return = 0
        end = perf_counter()

    for name in NETWORKS:
        if not all(torch.isfinite(p).all() for p in getattr(agent, name).parameters()):
            raise RuntimeError(f"Non-finite parameters in {name}")
    if not torch.isfinite(agent.log_alpha):
        raise RuntimeError("Non-finite SAC temperature")
    actor_delta = max((p.detach() - before).abs().max().item()
                      for p, before in zip(agent.actor.parameters(), initial_actor))
    if actor_delta <= 0:
        raise RuntimeError("SAC actor did not change")
    path = args.output / "sac_cpu.pt"
    torch.save({**checkpoint(agent), "run_config": config, "rng": {
        "torch": torch.get_rng_state(), "python": random.getstate(),
        "goals": goal_rng.bit_generator.state, "actions": action_rng.bit_generator.state,
    }}, path)
    original = deterministic_action(agent, reference)
    reloaded = deterministic_action(reload_agent(path), reference)
    np.testing.assert_array_equal(original, reloaded)
    summary = {"status": "PASS", "steps": args.steps, "updates": updates,
               "rollout_update_s": end - start, "update_phase_steps_per_s": updates / (end - update_start),
               "actor_max_parameter_change": actor_delta, "reload_action_max_abs_difference": 0.0,
               "checkpoint": str(path.resolve())}
    write_json(args.output / "summary.json", summary)
    return summary


def command_controller(observation):
    # Slew the inner position servo setpoint toward the joint goal.
    goal, ctrl = observation[4:6], observation[6:8]
    return np.clip((goal - ctrl) / ACTION_SCALE, -1, 1)


def run_episode(env, policy, goal, seed, delay, log):
    observation, _ = env.reset(goal=goal)
    initial = observation.copy()
    queue = deque(np.zeros(2) for _ in range(delay))
    episode_return = 0
    command_clips = 0
    while not env.finished:
        action = np.asarray(policy(observation), dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise RuntimeError("Expected a finite two-dimensional action in [-1, 1]")
        queue.append(action)
        requested = ACTION_SCALE * queue.popleft()
        next_obs, reward, terminated, truncated, result = env.step(requested)
        clipped = not np.allclose(requested, result["applied_increment_rad"], rtol=0, atol=1e-10)
        command_clips += int(clipped)
        json_line(log, {"seed": seed, "step": env.steps, "sim_time_s": float(env.data.time),
                        "observation": vector(observation), "next_observation": vector(next_obs),
                        "action_normalized": vector(action), "requested_rad": vector(requested),
                        "applied_rad": vector(result["applied_increment_rad"]),
                        "reward": reward, "terminated": terminated, "truncated": truncated,
                        "outcome": result["outcome"], "command_clipped": clipped,
                        "physics_warning": result["physics_warning"],
                        "actual_limit_violation": result["actual_limit_violation"]})
        episode_return += reward
        observation = next_obs
    return {"seed": seed, "goal_rad": goal, "initial_observation": vector(initial),
            "return": episode_return, "steps": env.steps, "outcome": result["outcome"],
            "final_abs_error_rad": vector(np.abs(observation[:2] - env.goal)),
            "final_abs_speed_rad_s": vector(np.abs(observation[2:4])),
            "command_clip_steps": command_clips, "physics_warning": result["physics_warning"],
            "actual_limit_violation": result["actual_limit_violation"]}


def evaluate(args):
    info = model_info(args.scene)
    policies = {"position_target": lambda rng: command_controller,
                "home_hold": lambda rng: lambda obs: np.zeros(2),
                "random": lambda rng: lambda obs: rng.uniform(-1, 1, 2)}
    if args.checkpoint is not None:
        import torch
        from panda_sac_smoke import reload_agent

        torch.set_num_threads(1)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        run = saved.get("run_config", {})
        if any(run.get(key) != info[key] for key in ("xml_sha256", "action_scale_rad", "max_episode_steps")):
            raise ValueError("Checkpoint must come from this training protocol with the same model XML")
        agent = reload_agent(args.checkpoint)
        policies["sac"] = lambda rng: lambda obs: deterministic_action(agent, obs)

    cases = EVAL_CASES[args.split]
    env = PandaTwoJointEnv(args.scene, max_steps=MAX_STEPS)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", {
        **info, "split": args.split, "cases": cases, "delay_steps": args.delay_steps,
        "position_error_threshold_rad": 0.01, "speed_threshold_rad_s": 0.05,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
    })
    summary = {}
    for name, make_policy in policies.items():
        with (args.output / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            episodes = [run_episode(env, make_policy(np.random.default_rng(seed)), goal,
                                    seed, args.delay_steps, stream) for seed, goal in cases]
        successes = [episode["steps"] for episode in episodes if episode["outcome"] == "success"]
        summary[name] = {"successes": len(successes), "episodes": len(cases),
                         "mean_success_steps": sum(successes) / len(successes) if successes else None,
                         "mean_return": sum(episode["return"] for episode in episodes) / len(cases),
                         "per_episode": episodes}
    write_json(args.output / "summary.json", summary)
    return {name: {key: value for key, value in report.items() if key != "per_episode"}
            for name, report in summary.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    training = commands.add_parser("train", help="Train on the fixed training goals")
    training.add_argument("--steps", type=int, required=True)
    training.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate", help="Evaluate baselines and optionally a checkpoint")
    evaluation.add_argument("--split", choices=EVAL_CASES, default="validation")
    evaluation.add_argument("--delay-steps", type=int, choices=range(11), default=0, metavar="0-10")
    evaluation.add_argument("--checkpoint", type=Path)
    for command in (training, evaluation):
        command.add_argument("scene", type=Path)
        command.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    result = train(args) if args.mode == "train" else evaluate(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
