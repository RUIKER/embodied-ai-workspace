"""Run checkpoint feedback control across ROS 2 services in Panda simulation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import secrets

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

from panda_policy_actor import actor_action, load_actor, model_digest
from panda_policy_interfaces.srv import ResetEpisode, StepEpisode
from panda_reach import (
    ACTION_SCALE, MAX_STEPS, command_clipped, json_line, model_info,
    normalized_action, run_episode, vector, write_json,
)
from panda_two_joint_env import PandaTwoJointEnv


RESET_SERVICE = "/panda_policy/reset_episode"
STEP_SERVICE = "/panda_policy/step_episode"
SERVICE_TIMEOUT_S = 5.0


class PolicySimulator(Node):
    def __init__(self, scene):
        super().__init__("panda_policy_sim")
        self.env = PandaTwoJointEnv(scene, max_steps=MAX_STEPS)
        self.digest = model_digest(scene)
        self.episode = 0
        self.state_publisher = self.create_publisher(JointState, "/panda_policy/joint_states", 10)
        self.create_service(ResetEpisode, RESET_SERVICE, self.reset_episode)
        self.create_service(StepEpisode, STEP_SERVICE, self.step_episode)

    def publish_state(self, observation):
        stamp = round(self.env.data.time * 1_000_000_000)
        message = JointState()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        message.name = ["joint1", "joint2"]
        message.position = observation[:2].tolist()
        message.velocity = observation[2:4].tolist()
        self.state_publisher.publish(message)

    def reset_episode(self, request, response):
        try:
            observation, _ = self.env.reset(goal=request.goal_rad)
        except ValueError as error:
            response.reason = str(error)
            return response
        self.episode = secrets.randbelow(2**64 - 1) + 1
        response.accepted = True
        response.episode = self.episode
        response.model_digest = self.digest
        response.observation = observation.tolist()
        self.publish_state(observation)
        return response

    def step_episode(self, request, response):
        if request.episode != self.episode:
            response.reason = "wrong_episode"
            return response
        if self.env.finished:
            response.reason = "episode_finished"
            return response
        if request.step != self.env.steps + 1:
            response.reason = "unexpected_step"
            return response
        try:
            observation, reward, terminated, truncated, info = self.env.step(request.increment_rad)
        except ValueError as error:
            response.reason = str(error)
            return response

        response.accepted = True
        response.episode = self.episode
        response.step = self.env.steps
        response.sim_time_s = float(self.env.data.time)
        response.observation = observation.tolist()
        response.applied_increment_rad = info["applied_increment_rad"].tolist()
        response.reward = reward
        response.terminated = terminated
        response.truncated = truncated
        response.outcome = info["outcome"]
        response.physics_warning = info["physics_warning"]
        response.actual_limit_violation = info["actual_limit_violation"]
        self.publish_state(observation)
        return response


def call(node, client, request):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=SERVICE_TIMEOUT_S)
    if not future.done():
        client.remove_pending_request(future)
        raise TimeoutError(f"No response from {client.srv_name}")
    return future.result()


def run_ros_episode(node, reset, step, digest, policy, case_id, goal, log, previous_episode):
    reset_response = call(node, reset, ResetEpisode.Request(goal_rad=goal))
    if not reset_response.accepted:
        raise RuntimeError(f"Reset rejected: {reset_response.reason}")
    if reset_response.model_digest != digest:
        raise RuntimeError("ROS simulator uses a different Panda model")
    episode = reset_response.episode
    if previous_episode is not None:
        old = call(node, step, StepEpisode.Request(episode=previous_episode, step=1))
        if old.accepted or old.reason != "wrong_episode":
            raise RuntimeError("Stale episode request was not rejected")
    observation = np.asarray(reset_response.observation)

    for index in range(1, MAX_STEPS + 1):
        action = normalized_action(np.asarray(policy(observation), dtype=float))
        requested = action * ACTION_SCALE
        request = StepEpisode.Request(episode=episode, step=index, increment_rad=requested.tolist())
        result = call(node, step, request)
        if not result.accepted or (result.episode, result.step) != (episode, index):
            raise RuntimeError(f"Step {index} rejected or mismatched: {result.reason}")

        json_line(log, {"seed": case_id, "step": index, "sim_time_s": result.sim_time_s,
                        "observation": vector(observation),
                        "next_observation": vector(result.observation),
                        "action_normalized": vector(action), "requested_rad": vector(requested),
                        "applied_rad": vector(result.applied_increment_rad),
                        "reward": result.reward, "terminated": result.terminated,
                        "truncated": result.truncated, "outcome": result.outcome,
                        "command_clipped": command_clipped(requested, result.applied_increment_rad),
                        "physics_warning": result.physics_warning,
                        "actual_limit_violation": result.actual_limit_violation})
        observation = np.asarray(result.observation)
        if result.terminated or result.truncated:
            after = call(node, step, StepEpisode.Request(episode=episode, step=index + 1))
            if after.accepted or after.reason != "episode_finished":
                raise RuntimeError("Step after episode end was not rejected")
            return episode
        if index == 1:
            duplicate = call(node, step, request)
            if duplicate.accepted or duplicate.reason != "unexpected_step":
                raise RuntimeError("Duplicate step was not rejected")
    raise RuntimeError("ROS simulator did not report the episode end")


def evaluate(node, args):
    weights, checkpoint_sha256 = load_actor(args.actor, args.scene)
    with args.protocol.open(encoding="utf-8") as stream:
        evaluation = json.load(stream)["evaluation"]
    if evaluation["delay_steps"] != 0 or evaluation["policy"] != "deterministic":
        raise ValueError("Expected a deterministic evaluation protocol without action delay")
    available = dict(evaluation["cases"])
    cases = [(case_id, available[case_id]) for case_id in args.case_ids]
    env = PandaTwoJointEnv(args.scene, max_steps=MAX_STEPS)
    reset = node.create_client(ResetEpisode, RESET_SERVICE)
    step = node.create_client(StepEpisode, STEP_SERVICE)
    for client in (reset, step):
        if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise TimeoutError(f"Service unavailable: {client.srv_name}")
    args.output.mkdir(parents=True, exist_ok=False)
    digest = model_digest(args.scene)
    write_json(args.output / "config.json", {
        **model_info(args.scene), "model_digest": digest, "checkpoint_sha256": checkpoint_sha256,
        "actor_sha256": sha256(args.actor.read_bytes()).hexdigest(),
        "split": evaluation["split"], "cases": cases, "delay_steps": 0,
    })

    def policy(observation):
        return actor_action(weights, observation)

    summaries = []
    previous_episode = None
    for case_id, goal in cases:
        offline_path = args.output / f"{case_id}-offline.jsonl"
        ros_path = args.output / f"{case_id}-ros2.jsonl"
        with offline_path.open("x", encoding="utf-8") as stream:
            result = run_episode(env, policy, goal, case_id, 0, stream)
        with ros_path.open("x", encoding="utf-8") as stream:
            previous_episode = run_ros_episode(
                node, reset, step, digest, policy, case_id, goal, stream, previous_episode,
            )
        offline_rows = [json.loads(line) for line in offline_path.read_text(encoding="utf-8").splitlines()]
        ros_rows = [json.loads(line) for line in ros_path.read_text(encoding="utf-8").splitlines()]
        if ros_rows != offline_rows:
            raise RuntimeError(f"ROS and offline feedback differ for case {case_id}")
        summaries.append(result)
    report = {"status": "PASS", "same_actor_step_log_matches": True,
              "duplicate_step_rejected": True, "post_episode_step_rejected": True,
              "stale_episode_rejected": len(summaries) > 1, "per_episode": summaries}
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("scene", type=Path)
    client = commands.add_parser("evaluate")
    client.add_argument("scene", type=Path)
    client.add_argument("actor", type=Path)
    client.add_argument("protocol", type=Path)
    client.add_argument("output", type=Path)
    client.add_argument("--case", dest="case_ids", type=int, action="append", required=True)
    args = parser.parse_args()
    if args.mode == "evaluate" and len(set(args.case_ids)) != len(args.case_ids):
        parser.error("Case IDs must be unique")

    try:
        with rclpy.init(args=[]):
            if args.mode == "serve":
                node = PolicySimulator(args.scene)
                try:
                    rclpy.spin(node)
                finally:
                    node.destroy_node()
            else:
                node = Node("panda_policy_evaluate")
                try:
                    evaluate(node, args)
                finally:
                    node.destroy_node()
    except (KeyboardInterrupt, ExternalShutdownException):
        if args.mode != "serve":
            raise


if __name__ == "__main__":
    main()
