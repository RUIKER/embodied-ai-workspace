"""Train Panda two-joint SAC or compare controllers on fixed reaching cases."""

import argparse
from collections import Counter, deque
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter

import mujoco
import numpy as np

from panda_her import PandaHERReplayBuffer
from panda_two_joint_env import (
    ACTION_INCREMENT_LIMIT_RAD,
    POSITION_ERROR_THRESHOLD_RAD,
    SPEED_THRESHOLD_RAD_S,
    PandaTwoJointEnv,
)


ACTION_SCALE = ACTION_INCREMENT_LIMIT_RAD
MAX_STEPS = 250
ACTION_SATURATION_THRESHOLD = 0.95
TRAIN_GOALS = [(x, y) for x in (-0.16, -0.08, 0.08, 0.16) for y in (-0.16, -0.08, 0.08, 0.16)]
DEMONSTRATION_GOALS = [goal for goal in TRAIN_GOALS if abs(goal[0]) == abs(goal[1])]
DEMONSTRATION_STEPS = 152
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


def normalized_action(action):
    action = np.asarray(action)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise RuntimeError("Expected a finite two-dimensional action in [-1, 1]")
    return action


def command_clipped(requested, applied):
    return not np.allclose(requested, applied, rtol=0, atol=1e-10)


def should_log(index, final_index, interval):
    return interval > 0 and (index in (1, final_index) or index % interval == 0)


class EpisodeLog:
    def __init__(self, initial_observation):
        self.initial_observation = initial_observation.copy()
        self.total_reward = 0.0
        self.actions = []
        self.clip_steps = 0
        self.minimum_error = float("inf")
        self.position_steps = self.speed_steps = self.settled_steps = 0

    def observe(self, action, reward, next_obs, clipped):
        self.total_reward += reward
        self.actions.append(action)
        self.clip_steps += int(clipped)
        error = np.max(np.abs(next_obs[:2] - next_obs[4:6]))
        speed = np.max(np.abs(next_obs[2:4]))
        if np.isfinite(error):
            self.minimum_error = min(self.minimum_error, float(error))
        position_ok = error < POSITION_ERROR_THRESHOLD_RAD
        speed_ok = speed < SPEED_THRESHOLD_RAD_S
        self.position_steps += int(position_ok)
        self.speed_steps += int(speed_ok)
        self.settled_steps += int(position_ok and speed_ok)

    def summary(self, next_obs):
        actions = np.asarray(self.actions)
        return {
            "initial_observation": vector(self.initial_observation),
            "return": self.total_reward,
            "min_max_abs_error_rad": self.minimum_error if np.isfinite(self.minimum_error) else None,
            "final_abs_error_rad": vector(np.abs(next_obs[:2] - next_obs[4:6])),
            "final_abs_speed_rad_s": vector(np.abs(next_obs[2:4])),
            "position_threshold_steps": self.position_steps,
            "speed_threshold_steps": self.speed_steps,
            "both_threshold_steps": self.settled_steps,
            "action_mean_normalized": vector(actions.mean(axis=0)),
            "action_std_normalized": vector(actions.std(axis=0)),
            "action_saturation_fraction_by_axis": vector(
                (np.abs(actions) >= ACTION_SATURATION_THRESHOLD).mean(axis=0)
            ),
            "command_clip_steps": self.clip_steps,
            "command_clip_fraction": self.clip_steps / len(actions),
        }


def train(args):
    import torch
    from panda_sac_smoke import CONFIG, NETWORKS, SAC_DIR, ReplayBuffer, SAC, checkpoint, reload_agent

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    goal_rng = np.random.default_rng(args.seed)
    action_rng = np.random.default_rng(args.seed + 1)
    warmup, batch, capacity = 256, 64, 100_000
    if args.steps <= warmup:
        raise ValueError(f"--steps must exceed the {warmup}-step warmup")
    demonstrations = deque(DEMONSTRATION_GOALS if args.demo_warmup else ())
    demonstration_steps = DEMONSTRATION_STEPS if args.demo_warmup else 0
    info = model_info(args.scene)
    env = PandaTwoJointEnv(args.scene, max_steps=MAX_STEPS)
    agent = SAC(**CONFIG)
    her_seed = args.seed + 2
    replay = (PandaHERReplayBuffer(capacity, env.lower, env.upper, her_seed)
              if args.her else ReplayBuffer(capacity))
    initial_actor = [p.detach().clone() for p in agent.actor.parameters()]
    script_dir = Path(__file__).resolve().parent
    sources = [
        Path(__file__), script_dir / "panda_two_joint_env.py", script_dir / "panda_sac_smoke.py",
        script_dir / "panda_her.py",
        *(SAC_DIR / name for name in ("agents.py", "networks.py", "rl_utils.py")),
    ]
    config = {**info, "steps": args.steps, "seed": args.seed, "action_rng_seed": args.seed + 1,
              "warmup": warmup, "batch": batch, "replay_capacity": capacity,
              "warmup_policy": {"demonstration_goals_rad": list(demonstrations),
                                "demonstration_steps": demonstration_steps,
                                "random_steps": warmup - demonstration_steps},
              "goals_rad": TRAIN_GOALS, "torch": str(torch.__version__), "sac": CONFIG,
              "reward_mode": "sparse", "her": {
                  "enabled": args.her, "goal_rng_seed": her_seed if args.her else None,
                  "strategy": "future_low_speed_inclusive" if args.her else None,
                  "attempt_fraction": 0.5 if args.her else 0.0,
              },
              "position_error_threshold_rad": POSITION_ERROR_THRESHOLD_RAD,
              "speed_threshold_rad_s": SPEED_THRESHOLD_RAD_S,
              "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in sources},
              "logging": {"step_log_every": args.step_log_every, "update_log_every": args.update_log_every,
                          "action_saturation_threshold": ACTION_SATURATION_THRESHOLD}}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", config)

    def reset():
        goal = demonstrations[0] if demonstrations else TRAIN_GOALS[goal_rng.integers(len(TRAIN_GOALS))]
        return env.reset(goal=goal)[0]

    observation = reset()
    reference = observation.copy()
    episode = EpisodeLog(observation)
    outcomes = Counter()
    her_totals = Counter()
    episode_index, updates = 1, 0
    with ExitStack() as stack:
        episode_log = stack.enter_context((args.output / "episodes.jsonl").open("x", encoding="utf-8"))
        step_log = (stack.enter_context((args.output / "steps.jsonl").open("x", encoding="utf-8"))
                    if args.step_log_every else None)
        update_log = (stack.enter_context((args.output / "updates.jsonl").open("x", encoding="utf-8"))
                      if args.update_log_every else None)
        start = perf_counter()
        for step in range(1, args.steps + 1):
            if step == warmup + 1:
                update_start = perf_counter()
            warmup_step = step <= warmup
            if demonstrations:
                source = "demonstration"
                action = command_controller(observation)
            elif warmup_step:
                source = "warmup"
                action = action_rng.uniform(-1, 1, 2)
            else:
                source = "actor"
                action = agent.take_action(observation)
            action = normalized_action(action)
            requested = ACTION_SCALE * action
            next_obs, reward, terminated, truncated, result = env.step(requested)
            applied = result["applied_increment_rad"]
            clipped = command_clipped(requested, applied)
            episode.observe(action, reward, next_obs, clipped)
            if step_log is not None and should_log(step, args.steps, args.step_log_every):
                json_line(step_log, {"step": step, "episode": episode_index, "episode_step": env.steps,
                                     "sim_time_s": float(env.data.time),
                                     "source": source,
                                     "observation": vector(observation), "next_observation": vector(next_obs),
                                     "action_normalized": vector(action), "requested_rad": vector(requested),
                                     "applied_rad": vector(applied), "command_clipped": clipped,
                                     "reward": reward, "terminated": terminated, "truncated": truncated,
                                     "outcome": result["outcome"], "physics_warning": result["physics_warning"],
                                     "actual_limit_violation": result["actual_limit_violation"]})
            if terminated or truncated or step == args.steps:
                outcome = result["outcome"] if terminated or truncated else "partial"
                outcomes[outcome] += 1
                json_line(episode_log, {"step": step, "episode": episode_index, "goal_rad": vector(env.goal),
                                       "episode_steps": env.steps, "terminated": terminated,
                                       "truncated": truncated, "outcome": outcome,
                                       "physics_warning": result["physics_warning"],
                                       "actual_limit_violation": result["actual_limit_violation"],
                                       **episode.summary(next_obs)})
            if not np.isfinite(next_obs).all() or result["outcome"] == "failure":
                raise RuntimeError(f"Aborting training on simulation failure at step {step}")
            if source == "demonstration":
                if truncated:
                    raise RuntimeError(f"Warmup demonstration timed out at step {step}")
                if terminated:
                    demonstrations.popleft()
                if bool(demonstrations) != (step < demonstration_steps):
                    raise RuntimeError(f"Warmup demonstrations must finish at step {demonstration_steps}")
            if args.her:
                replay.add(observation, action, reward, next_obs, terminated,
                           truncated=truncated, failed=False)
            else:
                replay.add(observation, action, reward, next_obs, terminated)
            if not warmup_step:
                updates += 1
                log_update = update_log is not None and should_log(
                    updates, args.steps - warmup, args.update_log_every
                )
                samples = replay.sample(batch)
                her_log = {}
                if args.her:
                    samples, counts = samples
                    her_totals.update(counts)
                    her_log["her"] = counts
                try:
                    metrics = agent.update(samples, diagnostics=log_update)
                except FloatingPointError as exc:
                    if update_log is not None:
                        json_line(update_log, {"step": step, "update": updates, "error": str(exc)})
                    raise
                if log_update:
                    json_line(update_log, {"step": step, "update": updates, "replay_size": replay.size(),
                                          **metrics, **her_log})
            observation = next_obs
            if terminated or truncated:
                observation = reset()
                episode = EpisodeLog(observation)
                episode_index += 1
        end = perf_counter()

    for name in NETWORKS:
        if not all(torch.isfinite(p).all() for p in getattr(agent, name).parameters()):
            raise RuntimeError(f"Non-finite parameters in {name}")
    final_alpha = agent.log_alpha.detach().exp().item()
    if not np.isfinite(final_alpha) or final_alpha <= 0:
        raise RuntimeError("Invalid SAC temperature")
    actor_delta = max((p.detach() - before).abs().max().item()
                      for p, before in zip(agent.actor.parameters(), initial_actor))
    if actor_delta <= 0:
        raise RuntimeError("SAC actor did not change")
    path = args.output / "sac_cpu.pt"
    torch.save({**checkpoint(agent), "run_config": config, "rng": {
        "torch": torch.get_rng_state(), "python": random.getstate(),
        "goals": goal_rng.bit_generator.state, "actions": action_rng.bit_generator.state,
        **({"her_goals": replay.goal_rng.getstate()} if args.her else {}),
    }}, path)
    original = deterministic_action(agent, reference)
    reloaded = deterministic_action(reload_agent(path), reference)
    np.testing.assert_array_equal(original, reloaded)
    summary = {"status": "PASS", "steps": args.steps, "updates": updates,
               "rollout_update_s": end - start, "update_phase_steps_per_s": updates / (end - update_start),
               "episode_outcomes": dict(outcomes), "final_alpha": final_alpha,
               "actor_max_parameter_change": actor_delta, "reload_action_max_abs_difference": 0.0,
               "checkpoint": str(path.resolve())}
    if args.her:
        summary["her"] = dict(her_totals)
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
        action = normalized_action(np.asarray(policy(observation), dtype=float))
        queue.append(action)
        requested = ACTION_SCALE * queue.popleft()
        next_obs, reward, terminated, truncated, result = env.step(requested)
        clipped = command_clipped(requested, result["applied_increment_rad"])
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
        "position_error_threshold_rad": POSITION_ERROR_THRESHOLD_RAD,
        "speed_threshold_rad_s": SPEED_THRESHOLD_RAD_S,
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


def log_interval(value):
    interval = int(value)
    if interval < 0:
        raise argparse.ArgumentTypeError("log interval must be nonnegative")
    return interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    training = commands.add_parser("train", help="Train on the fixed training goals")
    training.add_argument("--steps", type=int, required=True)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--her", action="store_true",
                          help="Relabel up to half of each batch with same-episode future low-speed goals")
    training.add_argument("--demo-warmup", action="store_true",
                          help="Use 152 position-controller steps and 104 random steps for warmup")
    training.add_argument("--step-log-every", type=log_interval, default=250,
                          help="Record a transition every N environment steps (0 disables)")
    training.add_argument("--update-log-every", type=log_interval, default=250,
                          help="Record losses and gradients every N updates (0 disables)")
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
