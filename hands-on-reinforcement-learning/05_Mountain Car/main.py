"""Mountain Car actor-critic 实验入口。"""

from pathlib import Path
import numpy as np

from config import (
    DISCOUNT_FACTOR, ENVIRONMENT_NAME, FEATURE_SAMPLES, NUM_EPISODES,
    POLICY_LEARNING_RATE, RANDOM_SEED, RBF_COMPONENTS, RBF_GAMMAS,
    SMOOTHING_WINDOW, VALUE_LEARNING_RATE,
)
from environment import make_environment
from estimators import GaussianPolicy, ValueEstimator
from features import RBFStateFeaturizer
from trainer import train
from visualization import plot_episode_stats


def main():
    """构建依赖并运行一次完整训练 pipeline。"""
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_environment(ENVIRONMENT_NAME, RANDOM_SEED)
    try:
        featurizer = RBFStateFeaturizer(
            env.observation_space, FEATURE_SAMPLES, RBF_COMPONENTS, RBF_GAMMAS,
            RANDOM_SEED,
        )
        feature_count = featurizer.feature_count
        policy = GaussianPolicy(
            feature_count, env.action_space.low[0], env.action_space.high[0],
            POLICY_LEARNING_RATE, seed=RANDOM_SEED,
        )
        value_estimator = ValueEstimator(feature_count, VALUE_LEARNING_RATE)
        stats = train(
            env, featurizer, policy, value_estimator, NUM_EPISODES,
            DISCOUNT_FACTOR,
        )
        np.savetxt(
            output_dir / "episode_stats.csv",
            np.column_stack((stats.episode_rewards, stats.episode_lengths)),
            delimiter=",", header="episode_reward,episode_length", comments="",
        )
        (output_dir / "run_config.txt").write_text(
            "environment=" + ENVIRONMENT_NAME + "\n"
            f"episodes={NUM_EPISODES}\n"
            f"discount_factor={DISCOUNT_FACTOR}\n"
            f"policy_learning_rate={POLICY_LEARNING_RATE}\n"
            f"value_learning_rate={VALUE_LEARNING_RATE}\n"
            f"random_seed={RANDOM_SEED}\n",
            encoding="utf-8",
        )
        plot_episode_stats(stats, SMOOTHING_WINDOW, output_dir, show=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()