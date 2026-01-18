"""
LEGACY: Curriculum learning V2 with NPCEnv (uses language embeddings).

For the current architecture, use `train_simple.py` instead, which:
- Uses SimpleNPCEnv (24-dim observation, no embeddings)
- Has two-phase reward training (exploration → precision)
- Inherits CurriculumV2 improvements

This file is kept for backward compatibility and reference.

Original V2 improvements over V1:
1. Warmup period - ignores early lucky episodes
2. Mixed-difficulty sampling - gradual shift instead of hard switches
3. Distance-progress focused rewards - less reliance on terminal success
4. Dynamic success radius - starts relaxed, tightens over time
"""
import argparse
from typing import Optional, List, Callable
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.training import NPCEnv
from src.training.curriculum_v2 import CurriculumV2Controller, CurriculumV2Config
from src.world.world import WorldConfig


class CurriculumV2Callback(BaseCallback):
    """
    Callback for Curriculum V2 with:
    - Mixed-difficulty world size sampling
    - Dynamic success radius
    - Warmup period before tracking
    """

    def __init__(
        self,
        curriculum: CurriculumV2Controller,
        envs: List,  # List of unwrapped NPCEnv instances
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.envs = envs  # Direct references to NPCEnv instances
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            if done:
                self._episode_count += 1

                # Check completion
                termination_reason = info.get("termination_reason", "unknown")
                completed = termination_reason == "INTENT_COMPLETED"

                # Get the world size that was used for this episode
                world_size = self.envs[i].world_size if i < len(self.envs) else 8

                # Record result
                self.curriculum.record_episode(completed, world_size)

                # Sample new world size for next episode and update env
                new_size = self.curriculum.sample_world_size()
                new_radius = self.curriculum.current_success_radius

                if i < len(self.envs):
                    self.envs[i].set_world_size(new_size)
                    self.envs[i].set_distance_threshold(new_radius)

                # Logging
                if self.verbose > 0:
                    # Debug first few episodes
                    if self._episode_count <= 5:
                        ep_info = info.get("episode", {})
                        print(f"[Debug] Episode {self._episode_count}: "
                              f"world={world_size}, "
                              f"reason={termination_reason}, "
                              f"reward={ep_info.get('r', 'N/A'):.2f}")

                    # Periodic stats
                    if self._episode_count % 50 == 0:
                        stats = self.curriculum.get_stats()
                        if stats["in_warmup"]:
                            print(f"[Curriculum] Episode {self._episode_count} "
                                  f"(warmup: {stats['warmup_remaining']} remaining)")
                        else:
                            weights = stats["sampling_weights"]
                            weights_str = " ".join(f"{k}:{v:.0%}" for k, v in weights.items())
                            print(f"[Curriculum] Episode {self._episode_count}: "
                                  f"success={stats['success_rate']:.1%}, "
                                  f"radius={stats['success_radius']:.1f}, "
                                  f"weights=[{weights_str}]")

        return True

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            stats = self.curriculum.get_stats()
            print(f"\n{'='*60}")
            print("CURRICULUM V2 TRAINING COMPLETE")
            print(f"  Total episodes: {stats['total_episodes']}")
            print(f"  Final success rate: {stats['success_rate']:.1%}")
            print(f"  Final success radius: {stats['success_radius']:.1f}")
            print(f"  Sampling weights: {stats['sampling_weights']}")
            print(f"  Per-size success rates: {stats['per_size_success']}")
            print(f"{'='*60}\n")


def make_env_v2(
    world_size: int,
    seed: int,
    rank: int = 0,
    log_dir: Optional[str] = None
) -> Callable:
    """Create environment factory that returns (Monitor, NPCEnv) for tracking."""
    def _init():
        env = NPCEnv(
            world_size=world_size,
            max_steps_per_episode=500,
            seed=seed + rank,
        )
        # Wrap with Monitor
        monitor_path = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            monitor_path = f"{log_dir}/monitor_{rank}"
        return Monitor(env, filename=monitor_path)
    return _init


def train_curriculum_v2(
    total_timesteps: int = 1_000_000,  # Increased for slower curriculum
    n_envs: int = 4,
    seed: int = 42,
    save_dir: str = "models/curriculum_v2",
    log_dir: str = "logs/curriculum_v2",
    # Curriculum V2.1 config - more conservative progression
    world_sizes: Optional[List[int]] = None,
    warmup_episodes: int = 100,
    initial_weights: Optional[List[float]] = None,
    initial_radius: float = 5.0,
    final_radius: float = 2.0,
    radius_tightening_episodes: int = 3000,  # Slower radius tightening
    shift_threshold: float = 0.8,  # Higher threshold to shift
    difficulty_shift_rate: float = 0.02,  # Slower shift
):
    """
    Train with Curriculum V2.
    """
    set_random_seed(seed)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set up curriculum
    if world_sizes is None:
        world_sizes = [8, 16, 32, 64]
    if initial_weights is None:
        initial_weights = [0.85, 0.10, 0.04, 0.01]  # V2.1: More concentrated on small

    curriculum_config = CurriculumV2Config(
        world_sizes=world_sizes,
        warmup_episodes=warmup_episodes,
        window_size=200,
        initial_weights=initial_weights,
        difficulty_shift_rate=difficulty_shift_rate,  # V2.1: Use parameter
        shift_threshold=shift_threshold,  # V2.1: Use parameter
        initial_success_radius=initial_radius,
        final_success_radius=final_radius,
        radius_tightening_episodes=radius_tightening_episodes,  # V2.1: Use parameter
    )
    curriculum = CurriculumV2Controller(curriculum_config)

    print("="*60)
    print("CURRICULUM V2.1 SETUP (Conservative Progression)")
    print("="*60)
    print(f"World sizes: {world_sizes}")
    print(f"Initial weights: {dict(zip(world_sizes, initial_weights))}")
    print(f"Warmup episodes: {warmup_episodes}")
    print(f"Success radius: {initial_radius} -> {final_radius} over {radius_tightening_episodes} episodes")
    print(f"Shift threshold: {shift_threshold:.0%} success rate (was 40%)")
    print(f"Shift rate: {difficulty_shift_rate} (was 0.1)")
    print("="*60)

    # Create environments
    initial_size = curriculum.sample_world_size()
    env_fns = [make_env_v2(initial_size, seed, i, log_dir) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)

    # Get references to the underlying NPCEnv instances (through Monitor wrapper)
    npc_envs = [env.envs[i].env for i in range(n_envs)]

    eval_env = DummyVecEnv([make_env_v2(initial_size, seed + 1000, 0, log_dir)])

    print(f"Using DummyVecEnv with {n_envs} environments")

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,  # Higher entropy for exploration
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )

    # Set up callbacks
    curriculum_callback = CurriculumV2Callback(
        curriculum=curriculum,
        envs=npc_envs,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best",
        log_path=log_dir,
        eval_freq=5000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("Reward focus: distance-progress (getting closer = good)")
    print("")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{save_dir}/final_model"
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    # Cleanup
    env.close()
    eval_env.close()

    return curriculum.get_stats()


def main():
    parser = argparse.ArgumentParser(description="Train with Curriculum V2.1 (Conservative)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup episodes before tracking success")
    parser.add_argument("--initial-radius", type=float, default=5.0,
                        help="Initial success radius (relaxed)")
    parser.add_argument("--final-radius", type=float, default=2.0,
                        help="Final success radius (tight)")
    parser.add_argument("--radius-episodes", type=int, default=3000,
                        help="Episodes to tighten radius (default: 3000)")
    parser.add_argument("--shift-threshold", type=float, default=0.8,
                        help="Success rate to trigger difficulty shift (default: 0.8)")
    parser.add_argument("--shift-rate", type=float, default=0.02,
                        help="How fast to shift difficulty (default: 0.02)")
    parser.add_argument("--world-sizes", type=str, default="8,16,32,64",
                        help="Comma-separated world sizes")
    parser.add_argument("--weights", type=str, default="0.85,0.10,0.04,0.01",
                        help="Comma-separated initial sampling weights")

    args = parser.parse_args()

    world_sizes = [int(x) for x in args.world_sizes.split(",")]
    weights = [float(x) for x in args.weights.split(",")]

    stats = train_curriculum_v2(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        warmup_episodes=args.warmup,
        initial_radius=args.initial_radius,
        final_radius=args.final_radius,
        radius_tightening_episodes=args.radius_episodes,
        shift_threshold=args.shift_threshold,
        difficulty_shift_rate=args.shift_rate,
        world_sizes=world_sizes,
        initial_weights=weights,
    )

    print("\nFinal Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
