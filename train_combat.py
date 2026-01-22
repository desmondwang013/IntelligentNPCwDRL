#!/usr/bin/env python3
"""
Training script for the combat RL agent.

Uses CombatEnv to train an agent that can fight enemies.
Supports different combat styles and curriculum learning.
"""
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.combat import CombatEnv, CombatRewardConfig
from src.world import WorldConfig
from src.world.entities import CombatStyle


class CombatCurriculumCallback(BaseCallback):
    """
    Callback to implement curriculum learning for combat.

    Curriculum stages:
    1. 1 enemy, exploration phase (reduced penalties)
    2. 1 enemy, precision phase (full penalties)
    3. 2 enemies, precision phase
    4. 3+ enemies, precision phase
    """

    def __init__(
        self,
        envs: list,
        phase1_episodes: int = 2000,
        phase2_episodes: int = 4000,
        phase3_episodes: int = 6000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.envs = envs
        self.phase1_episodes = phase1_episodes
        self.phase2_episodes = phase2_episodes
        self.phase3_episodes = phase3_episodes

        self._episode_count = 0
        self._current_phase = 1
        self._recent_wins = []
        self._window_size = 100

    def _on_step(self) -> bool:
        # Track episodes
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                self._episode_count += 1

                # Track wins
                enemies_alive = info.get("enemies_alive", 0)
                npc_alive = info.get("npc_alive", True)
                won = (enemies_alive == 0) and npc_alive

                self._recent_wins.append(won)
                if len(self._recent_wins) > self._window_size:
                    self._recent_wins.pop(0)

        # Check phase transitions
        self._check_phase_transition()

        return True

    def _check_phase_transition(self):
        """Check if we should advance to next phase."""
        win_rate = (
            sum(self._recent_wins) / len(self._recent_wins)
            if self._recent_wins else 0
        )

        if self._current_phase == 1 and self._episode_count >= self.phase1_episodes:
            if win_rate >= 0.5 or self._episode_count >= self.phase1_episodes * 1.5:
                self._transition_to_phase(2)

        elif self._current_phase == 2 and self._episode_count >= self.phase2_episodes:
            if win_rate >= 0.6 or self._episode_count >= self.phase2_episodes * 1.5:
                self._transition_to_phase(3)

        elif self._current_phase == 3 and self._episode_count >= self.phase3_episodes:
            if win_rate >= 0.5 or self._episode_count >= self.phase3_episodes * 1.5:
                self._transition_to_phase(4)

    def _transition_to_phase(self, phase: int):
        """Transition to a new curriculum phase."""
        print(f"\n{'='*60}")
        print(f"CURRICULUM: Phase {self._current_phase} -> Phase {phase}")
        print(f"Episodes: {self._episode_count}")
        print(f"Win rate: {sum(self._recent_wins)/len(self._recent_wins)*100:.1f}%")

        self._current_phase = phase

        for env in self.envs:
            if phase == 2:
                # Full penalties, still 1 enemy
                env.set_reward_phase(2)
                print("  - Switched to full penalty rewards")

            elif phase == 3:
                # 2 enemies
                env.set_num_enemies(2)
                print("  - Increased to 2 enemies")

            elif phase == 4:
                # 3 enemies
                env.set_num_enemies(3)
                print("  - Increased to 3 enemies")

        print(f"{'='*60}\n")

    @property
    def current_phase(self) -> int:
        return self._current_phase

    @property
    def win_rate(self) -> float:
        return sum(self._recent_wins) / len(self._recent_wins) if self._recent_wins else 0


def make_combat_env(
    world_size: int = 16,
    num_enemies: int = 1,
    combat_style: CombatStyle = CombatStyle.BALANCED,
    max_steps: int = 500,
    seed: int = 42,
    reward_phase: int = 1,
):
    """Factory function to create a CombatEnv."""
    world_config = WorldConfig(size=world_size, num_objects=0)

    reward_config = CombatRewardConfig(phase=reward_phase)

    env = CombatEnv(
        world_config=world_config,
        num_enemies=num_enemies,
        combat_style=combat_style,
        max_steps=max_steps,
        reward_config=reward_config,
        seed=seed,
    )
    return env


def main():
    parser = argparse.ArgumentParser(description="Train combat RL agent")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluation frequency")
    parser.add_argument("--log-dir", type=str, default="logs/combat",
                        help="Log directory")
    parser.add_argument("--model-dir", type=str, default="models/combat",
                        help="Model save directory")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy coefficient (exploration)")
    parser.add_argument("--num-enemies", type=int, default=1,
                        help="Initial number of enemies")
    parser.add_argument("--world-size", type=int, default=16,
                        help="World size")
    parser.add_argument("--combat-style", type=str, default="balanced",
                        choices=["aggressive", "balanced", "passive"],
                        help="Combat style to train")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--phase1-episodes", type=int, default=2000,
                        help="Episodes for phase 1 (exploration)")
    parser.add_argument("--phase2-episodes", type=int, default=4000,
                        help="Episodes for phase 2 (precision, 1 enemy)")
    parser.add_argument("--phase3-episodes", type=int, default=6000,
                        help="Episodes for phase 3 (2 enemies)")

    args = parser.parse_args()

    # Create directories
    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Parse combat style
    style_map = {
        "aggressive": CombatStyle.AGGRESSIVE,
        "balanced": CombatStyle.BALANCED,
        "passive": CombatStyle.PASSIVE,
    }
    combat_style = style_map[args.combat_style]

    print("=" * 60)
    print("COMBAT RL AGENT TRAINING")
    print("=" * 60)
    print(f"Observation: 29 dimensions")
    print(f"Actions: 6 (4 movement + wait + attack)")
    print(f"Combat style: {args.combat_style}")
    print(f"Initial enemies: {args.num_enemies}")
    print(f"World size: {args.world_size}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Learning rate: {args.lr}")
    print(f"Entropy coef: {args.ent_coef}")
    print(f"Curriculum: {'Disabled' if args.no_curriculum else 'Enabled'}")
    print("=" * 60)

    # Create environments
    raw_envs = []
    for i in range(args.n_envs):
        env = make_combat_env(
            world_size=args.world_size,
            num_enemies=args.num_enemies,
            combat_style=combat_style,
            max_steps=500,
            seed=42 + i,
            reward_phase=1,  # Start with exploration phase
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
    eval_env = make_combat_env(
        world_size=args.world_size,
        num_enemies=1,  # Evaluate on 1 enemy for consistency
        combat_style=combat_style,
        max_steps=500,
        seed=999,
        reward_phase=2,  # Full penalties for eval
    )
    eval_env = Monitor(eval_env, str(log_dir / "eval"))

    print(f"\nObservation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Create PPO model
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
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
    curriculum_callback = None
    if not args.no_curriculum:
        curriculum_callback = CombatCurriculumCallback(
            envs=raw_envs,
            phase1_episodes=args.phase1_episodes,
            phase2_episodes=args.phase2_episodes,
            phase3_episodes=args.phase3_episodes,
            verbose=1,
        )
        callbacks.append(curriculum_callback)

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

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration}")
    if curriculum_callback:
        print(f"Final phase: {curriculum_callback.current_phase}")
        print(f"Final win rate: {curriculum_callback.win_rate*100:.1f}%")
    print("=" * 60)

    # Save final model
    final_path = model_dir / f"combat_{args.combat_style}_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}")

    # Cleanup
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
