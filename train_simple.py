#!/usr/bin/env python3
"""
Training script for the simplified RL executor.

Uses SimpleNPCEnv with 24-dimensional observation (no language embeddings).
This aligns with the architecture where RL handles motor execution only.
"""
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.training import SimpleNPCEnv
from src.training.curriculum_v2 import CurriculumV2Controller, CurriculumV2Config
from src.reward import RewardConfig


class CurriculumCallback(BaseCallback):
    """Callback to update curriculum based on training progress."""

    def __init__(
        self,
        curriculum: CurriculumV2Controller,
        envs: list,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.envs = envs
        self._episode_count = 0

    def _on_step(self) -> bool:
        # Check for episode completions
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self._episode_count += 1
                # Determine success
                termination = info.get("termination_reason", "")
                success = termination == "INTENT_COMPLETED"

                # Get world size for this env
                world_size = self.envs[i].world_size

                # Record episode
                self.curriculum.record_episode(success, world_size)

                # Sample new world size
                new_size = self.curriculum.sample_world_size()

                # Update environment
                self.envs[i].set_world_size(new_size)

                # Update success radius
                new_radius = self.curriculum.current_success_radius
                self.envs[i].set_distance_threshold(new_radius)

        return True


class RewardPhaseCallback(BaseCallback):
    """
    Callback to transition from exploration phase to precision phase.

    Two-phase training approach:
    - Exploration phase: No penalties, agent learns "movement = good"
    - Precision phase: Penalties added to refine behavior

    Transition happens when:
    - Episode count reaches threshold, OR
    - Success rate exceeds threshold
    """

    def __init__(
        self,
        envs: list,
        exploration_episodes: int = 2000,
        success_rate_threshold: float = 0.5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.envs = envs
        self.exploration_episodes = exploration_episodes
        self.success_rate_threshold = success_rate_threshold

        self._episode_count = 0
        self._recent_successes = []
        self._window_size = 200
        self._in_precision_phase = False

    def _on_step(self) -> bool:
        if self._in_precision_phase:
            return True

        # Track episodes and successes
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                self._episode_count += 1
                termination = info.get("termination_reason", "")
                success = termination == "INTENT_COMPLETED"
                self._recent_successes.append(success)

                if len(self._recent_successes) > self._window_size:
                    self._recent_successes.pop(0)

        # Check transition conditions
        should_transition = False

        # Condition 1: Episode threshold
        if self._episode_count >= self.exploration_episodes:
            should_transition = True
            reason = f"episode count ({self._episode_count})"

        # Condition 2: Success rate threshold
        if len(self._recent_successes) >= self._window_size:
            success_rate = sum(self._recent_successes) / len(self._recent_successes)
            if success_rate >= self.success_rate_threshold:
                should_transition = True
                reason = f"success rate ({success_rate:.1%})"

        if should_transition:
            self._transition_to_precision(reason)

        return True

    def _transition_to_precision(self, reason: str) -> None:
        """Switch all environments to precision phase rewards."""
        print(f"\n{'='*60}")
        print(f"PHASE TRANSITION: Exploration -> Precision")
        print(f"Triggered by: {reason}")
        print(f"Episodes completed: {self._episode_count}")
        print(f"{'='*60}\n")

        precision_config = RewardConfig.precision_phase()

        for env in self.envs:
            env.set_reward_config(precision_config)

        self._in_precision_phase = True

    @property
    def current_phase(self) -> str:
        return "precision" if self._in_precision_phase else "exploration"


def make_env(world_size: int, distance_threshold: float, seed: int):
    """Factory function to create a monitored SimpleNPCEnv."""
    def _init():
        env = SimpleNPCEnv(
            world_size=world_size,
            max_steps_per_episode=500,
            seed=seed,
            distance_threshold=distance_threshold,
        )
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train simplified RL executor")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Evaluation frequency")
    parser.add_argument("--log-dir", type=str, default="logs/simple",
                        help="Log directory")
    parser.add_argument("--model-dir", type=str, default="models/simple",
                        help="Model save directory")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy coefficient (exploration)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--exploration-episodes", type=int, default=2000,
                        help="Episodes in exploration phase before transition")
    parser.add_argument("--phase-success-threshold", type=float, default=0.5,
                        help="Success rate to trigger phase transition")
    parser.add_argument("--no-phased-rewards", action="store_true",
                        help="Disable phased reward training")

    args = parser.parse_args()

    # Create directories
    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine initial reward config
    use_phased_rewards = not args.no_phased_rewards
    if use_phased_rewards:
        initial_reward_config = RewardConfig.exploration_phase()
        reward_phase = "Exploration (no penalties)"
    else:
        initial_reward_config = None  # Use default
        reward_phase = "Default (all penalties)"

    print("=" * 60)
    print("SIMPLIFIED RL EXECUTOR TRAINING")
    print("=" * 60)
    print(f"Observation: 24 dimensions (NO language embeddings)")
    print(f"Architecture: RL receives structured goals only")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Learning rate: {args.lr}")
    print(f"Entropy coef: {args.ent_coef}")
    print(f"Curriculum: {'Disabled' if args.no_curriculum else 'Enabled'}")
    print(f"Phased rewards: {'Disabled' if args.no_phased_rewards else 'Enabled'}")
    print(f"Initial reward phase: {reward_phase}")
    if use_phased_rewards:
        print(f"  -> Exploration episodes: {args.exploration_episodes}")
        print(f"  -> Phase transition at: {args.phase_success_threshold:.0%} success")
    print("=" * 60)

    # Curriculum config
    curriculum_config = CurriculumV2Config(
        world_sizes=[8, 16, 32, 64],
        warmup_episodes=100,
        window_size=200,
        initial_weights=[0.85, 0.10, 0.04, 0.01],
        difficulty_shift_rate=0.02,
        shift_threshold=0.8,
        initial_success_radius=5.0,
        final_success_radius=2.0,
        radius_tightening_episodes=3000,
    )

    curriculum = CurriculumV2Controller(config=curriculum_config)

    # Initial settings
    if args.no_curriculum:
        initial_world_size = 64
        initial_radius = 2.0
    else:
        initial_world_size = curriculum.sample_world_size()
        initial_radius = curriculum.current_success_radius

    # Create environments
    raw_envs = []
    for i in range(args.n_envs):
        env = SimpleNPCEnv(
            world_size=initial_world_size,
            max_steps_per_episode=500,
            seed=42 + i,
            distance_threshold=initial_radius,
            reward_config=initial_reward_config,
        )
        raw_envs.append(env)

    # Wrap with Monitor
    monitored_envs = []
    for i, env in enumerate(raw_envs):
        monitored = Monitor(env, str(log_dir / f"monitor_{i}"))
        monitored_envs.append(monitored)

    # Create vectorized environment
    vec_env = DummyVecEnv([lambda e=env: e for env in monitored_envs])

    # Create eval environment
    eval_env = SimpleNPCEnv(
        world_size=8,  # Evaluate on small world for consistency
        max_steps_per_episode=500,
        seed=999,
        distance_threshold=2.0,
    )
    eval_env = Monitor(eval_env, str(log_dir / "eval"))

    print(f"\nObservation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Create PPO model
    # Smaller network since observation is much simpler
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=1,
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir),
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # Curriculum callback (if enabled)
    if not args.no_curriculum:
        curriculum_callback = CurriculumCallback(
            curriculum=curriculum,
            envs=raw_envs,
            verbose=1,
        )
        callbacks.append(curriculum_callback)

    # Reward phase callback (if enabled)
    reward_phase_callback = None
    if use_phased_rewards:
        reward_phase_callback = RewardPhaseCallback(
            envs=raw_envs,
            exploration_episodes=args.exploration_episodes,
            success_rate_threshold=args.phase_success_threshold,
            verbose=1,
        )
        callbacks.append(reward_phase_callback)

    # Train
    print("\nStarting training...\n")
    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Final stats
    duration = datetime.now() - start_time
    stats = curriculum.get_stats()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Final success rate: {stats['success_rate']*100:.1f}%")
    print(f"Final success radius: {stats['success_radius']}")
    print(f"Sampling weights: {stats['sampling_weights']}")
    print(f"Per-size success: {stats['per_size_success']}")
    if reward_phase_callback:
        print(f"Final reward phase: {reward_phase_callback.current_phase}")
    print("=" * 60)

    # Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}")

    # Save curriculum stats
    stats_path = log_dir / "curriculum_stats.txt"
    with open(stats_path, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"Curriculum stats saved to {stats_path}")

    # Cleanup
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
