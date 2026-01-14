"""
Training script for the NPC agent.

Usage:
    python train.py                    # Train with defaults
    python train.py --timesteps 50000  # Short training run
    python train.py --eval-only        # Evaluate existing model
"""
import argparse
from src.training import Trainer, TrainerConfig
from src.runtime import RuntimeConfig
from src.reward import RewardConfig


def main():
    parser = argparse.ArgumentParser(description="Train NPC agent with PPO")
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--save-dir", type=str, default="models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for tensorboard logs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate an existing model"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model to load (for eval or continued training)"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bar"
    )

    args = parser.parse_args()

    # Configure reward (can adjust these for tuning)
    reward_config = RewardConfig(
        progress_scale=1.0,
        completion_bonus=10.0,
        timeout_penalty=-5.0,
        time_penalty=-0.01,
        collision_penalty=-0.1,
        oscillation_penalty=-0.05,
    )

    # Configure runtime
    runtime_config = RuntimeConfig(
        default_intent_timeout=480,  # 30 seconds at 16 ticks/sec
        reward_config=reward_config,
    )

    # Configure trainer
    trainer_config = TrainerConfig(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        runtime_config=runtime_config,
        reward_config=reward_config,
        # PPO hyperparameters (defaults are reasonable for start)
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )

    # Create trainer
    trainer = Trainer(config=trainer_config)

    # Load existing model if specified
    if args.model_path:
        trainer.load(args.model_path)

    if args.eval_only:
        # Evaluation only
        if not args.model_path:
            print("Error: --model-path required for --eval-only")
            return

        print("\n=== Evaluating Model ===")
        results = trainer.evaluate(n_episodes=20, render=False)
        print(f"Mean reward: {results['mean_reward']:.2f} (+/- {results['std_reward']:.2f})")
        print(f"Mean episode length: {results['mean_length']:.1f}")

    else:
        # Training
        print("\n=== Training Configuration ===")
        print(f"Timesteps: {args.timesteps}")
        print(f"Parallel envs: {args.n_envs}")
        print(f"Seed: {args.seed}")
        print()

        # Train
        stats = trainer.train(progress_bar=not args.no_progress)

        # Save final model
        trainer.save("ppo_npc_final")

        # Final evaluation
        print("\n=== Final Evaluation ===")
        results = trainer.evaluate(n_episodes=20)
        print(f"Mean reward: {results['mean_reward']:.2f} (+/- {results['std_reward']:.2f})")
        print(f"Mean episode length: {results['mean_length']:.1f}")

        # Print training stats
        if stats["episode_rewards"]:
            print(f"\nTraining episodes completed: {len(stats['episode_rewards'])}")
            last_rewards = stats["episode_rewards"][-100:]
            print(f"Average reward (last 100): {sum(last_rewards)/len(last_rewards):.2f}")

    trainer.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
