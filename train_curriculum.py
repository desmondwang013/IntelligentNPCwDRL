"""
LEGACY: Curriculum learning V1 with NPCEnv (uses language embeddings).

For the current architecture, use `train_simple.py` instead, which:
- Uses SimpleNPCEnv (24-dim observation, no embeddings)
- Has two-phase reward training (exploration → precision)
- Uses CurriculumV2 with improved warmup and sampling

This file is kept for backward compatibility and reference.

Original description:
Train navigation with curriculum learning.
Starts with small world (8x8), gradually increases to full size (64x64)
as the agent demonstrates mastery at each level.
"""
import argparse
from dataclasses import dataclass
from typing import Optional, List, Callable
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.training import NPCEnv, CurriculumController, CurriculumConfig


class CurriculumCallback(BaseCallback):
    """
    Callback that tracks episode completions and manages curriculum progression.

    Updates world size across all environments when agent achieves
    sufficient success rate at current level.
    """

    def __init__(
        self,
        curriculum: CurriculumController,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.env = env
        self.eval_env = eval_env

        # Track episodes
        self._episode_count = 0
        self._completions_at_level = 0

    def _on_step(self) -> bool:
        # Check for completed episodes in infos
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, (info, done) in enumerate(zip(infos, dones)):
            # Episode ends when done is True
            if done:
                self._episode_count += 1

                # Check if this episode was a success (intent completed)
                # termination_reason is set by NPCEnv in step()
                termination_reason = info.get("termination_reason", "unknown")
                completed = termination_reason == "INTENT_COMPLETED"
                self.curriculum.record_episode(completed)

                if completed:
                    self._completions_at_level += 1

                # Debug: log first few episodes
                if self._episode_count <= 5 and self.verbose > 0:
                    ep_info = info.get("episode", {})
                    print(f"[Debug] Episode {self._episode_count}: "
                          f"reason={termination_reason}, "
                          f"reward={ep_info.get('r', 'N/A'):.2f}, "
                          f"length={ep_info.get('l', 'N/A')}")

                # Check for level advancement
                if self.curriculum.should_advance():
                    old_size = self.curriculum.current_world_size
                    self.curriculum.advance()
                    new_size = self.curriculum.current_world_size

                    if self.verbose > 0:
                        stats = self.curriculum.get_stats()
                        print(f"\n{'='*60}")
                        print(f"CURRICULUM ADVANCEMENT!")
                        print(f"  World size: {old_size} -> {new_size}")
                        print(f"  Success rate was: {stats['success_rate']:.1%}")
                        print(f"  Total episodes: {stats['total_episodes']}")
                        print(f"{'='*60}\n")

                    # Update all environments
                    self._update_env_world_size(new_size)
                    self._completions_at_level = 0

                # Periodic logging
                if self.verbose > 0 and self._episode_count % 50 == 0:
                    stats = self.curriculum.get_stats()
                    print(f"[Curriculum] Episodes: {self._episode_count}, "
                          f"Level: {stats['level']}, "
                          f"World: {stats['world_size']}x{stats['world_size']}, "
                          f"Success: {stats['success_rate']:.1%}")

        return True

    def _update_env_world_size(self, new_size: int) -> None:
        """Update world size on all environments."""
        # For DummyVecEnv with Monitor wrapper:
        # self.env.envs[i] is Monitor, self.env.envs[i].env is NPCEnv
        for i in range(len(self.env.envs)):
            monitor = self.env.envs[i]
            if hasattr(monitor, 'env'):
                # It's wrapped in Monitor
                monitor.env.set_world_size(new_size)
            else:
                # Direct NPCEnv (no wrapper)
                monitor.set_world_size(new_size)

        # Also update eval env if provided
        if self.eval_env is not None:
            for i in range(len(self.eval_env.envs)):
                monitor = self.eval_env.envs[i]
                if hasattr(monitor, 'env'):
                    monitor.env.set_world_size(new_size)
                else:
                    monitor.set_world_size(new_size)

    def _on_training_end(self) -> None:
        """Log final curriculum stats."""
        if self.verbose > 0:
            stats = self.curriculum.get_stats()
            print(f"\n{'='*60}")
            print("CURRICULUM TRAINING COMPLETE")
            print(f"  Final level: {stats['level']}")
            print(f"  Final world size: {stats['world_size']}x{stats['world_size']}")
            print(f"  Final success rate: {stats['success_rate']:.1%}")
            print(f"  Total episodes: {stats['total_episodes']}")
            print(f"  Reached max level: {stats['is_max_level']}")
            print(f"{'='*60}\n")


def make_env(world_size: int, seed: int, rank: int = 0, log_dir: Optional[str] = None) -> Callable:
    """Create an environment factory with Monitor wrapper for episode tracking."""
    def _init() -> Monitor:
        env = NPCEnv(
            world_size=world_size,
            max_steps_per_episode=500,
            seed=seed + rank,
        )
        # Wrap with Monitor to track episode statistics
        # This adds "episode" key to info dict on episode end
        monitor_path = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            monitor_path = f"{log_dir}/monitor_{rank}"
        return Monitor(env, filename=monitor_path)
    return _init


def train_with_curriculum(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    seed: int = 42,
    save_dir: str = "models/curriculum",
    log_dir: str = "logs/curriculum",
    # Curriculum config
    world_sizes: Optional[List[int]] = None,
    advance_threshold: float = 0.7,
    min_episodes: int = 50,
):
    """
    Train navigation using curriculum learning.

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        seed: Random seed
        save_dir: Directory for model checkpoints
        log_dir: Directory for logs
        world_sizes: Progression of world sizes (default: [8, 16, 32, 64])
        advance_threshold: Success rate needed to advance (0.0-1.0)
        min_episodes: Minimum episodes at each level before advancing
    """
    set_random_seed(seed)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set up curriculum
    if world_sizes is None:
        world_sizes = [8, 16, 32, 64]

    curriculum_config = CurriculumConfig(
        world_sizes=world_sizes,
        advance_threshold=advance_threshold,
        min_episodes_per_level=min_episodes,
        window_size=100,
    )
    curriculum = CurriculumController(curriculum_config)

    print("="*60)
    print("CURRICULUM LEARNING SETUP")
    print("="*60)
    print(f"World size progression: {world_sizes}")
    print(f"Advance threshold: {advance_threshold:.0%} success rate")
    print(f"Min episodes per level: {min_episodes}")
    print(f"Starting world size: {curriculum.current_world_size}x{curriculum.current_world_size}")
    print("="*60)

    # Create environments starting at smallest world size
    initial_size = curriculum.current_world_size

    # Use DummyVecEnv for curriculum learning
    # SubprocVecEnv doesn't support runtime world size changes
    env_fns = [make_env(initial_size, seed, i, log_dir) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    print(f"Using DummyVecEnv with {n_envs} environments (required for curriculum)")

    eval_env = DummyVecEnv([make_env(initial_size, seed + 1000, 0, log_dir)])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,  # Increased from 0.01 to encourage more exploration
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )

    # Set up callbacks
    curriculum_callback = CurriculumCallback(
        curriculum=curriculum,
        env=env,
        eval_env=eval_env,
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
    parser = argparse.ArgumentParser(description="Train with curriculum learning")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Success rate to advance (0.0-1.0)")
    parser.add_argument("--min-episodes", type=int, default=50,
                        help="Minimum episodes at each level")
    parser.add_argument("--world-sizes", type=str, default="8,16,32,64",
                        help="Comma-separated world size progression")

    args = parser.parse_args()

    world_sizes = [int(x) for x in args.world_sizes.split(",")]

    stats = train_with_curriculum(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        advance_threshold=args.threshold,
        min_episodes=args.min_episodes,
        world_sizes=world_sizes,
    )

    print("\nFinal Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
