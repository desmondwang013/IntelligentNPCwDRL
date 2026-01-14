"""
Training script for the dual policy architecture.

Usage:
    python train_dual_policy.py
    python train_dual_policy.py --target-steps 100000 --movement-steps 200000
    python train_dual_policy.py --eval-only --model-name my_model
"""
import argparse
from src.training import DualPolicyTrainer, DualTrainerConfig


def main():
    parser = argparse.ArgumentParser(description="Train dual policy NPC")

    # Training parameters
    parser.add_argument(
        "--target-steps", type=int, default=50000,
        help="Training timesteps for target selection policy"
    )
    parser.add_argument(
        "--movement-steps", type=int, default=100000,
        help="Training timesteps for movement policy"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments"
    )

    # Model management
    parser.add_argument(
        "--model-name", type=str, default="dual_policy",
        help="Name for saving/loading models"
    )
    parser.add_argument(
        "--save-dir", type=str, default="models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for tensorboard logs"
    )

    # Modes
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--target-only", action="store_true",
        help="Only train target selection policy"
    )
    parser.add_argument(
        "--movement-only", action="store_true",
        help="Only train movement policy"
    )
    parser.add_argument(
        "--no-progress-bar", action="store_true",
        help="Disable progress bar"
    )

    args = parser.parse_args()

    # Create config
    config = DualTrainerConfig(
        target_timesteps=args.target_steps,
        movement_timesteps=args.movement_steps,
        target_n_envs=args.n_envs,
        movement_n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )

    trainer = DualPolicyTrainer(config)

    if args.eval_only:
        # Load and evaluate
        print(f"Loading models: {args.model_name}")
        trainer.load(args.model_name)

        print("\nEvaluating target selection...")
        target_results = trainer.evaluate_target_selection(n_episodes=100)
        print(f"  Accuracy: {target_results['accuracy']*100:.1f}%")

        print("\nEvaluating movement...")
        movement_results = trainer.evaluate_movement(n_episodes=20)
        print(f"  Success rate: {movement_results['success_rate']*100:.1f}%")
        print(f"  Avg steps: {movement_results['avg_steps']:.1f}")

        print("\nEvaluating combined pipeline...")
        combined_results = trainer.evaluate_combined(n_episodes=20)
        print(f"  Target accuracy: {combined_results['target_accuracy']*100:.1f}%")
        print(f"  Navigation success: {combined_results['navigation_success']*100:.1f}%")
        print(f"  Full success: {combined_results['full_success']*100:.1f}%")

    else:
        # Train
        progress_bar = not args.no_progress_bar

        if args.target_only:
            trainer.train_target_selection(progress_bar=progress_bar)
        elif args.movement_only:
            trainer.train_movement(progress_bar=progress_bar)
        else:
            trainer.train_both(progress_bar=progress_bar)

        # Save
        trainer.save(args.model_name)

        # Quick evaluation
        print("\n" + "="*50)
        print("Quick Evaluation")
        print("="*50)

        if not args.movement_only:
            target_results = trainer.evaluate_target_selection(n_episodes=50)
            print(f"\nTarget Selection Accuracy: {target_results['accuracy']*100:.1f}%")

        if not args.target_only:
            movement_results = trainer.evaluate_movement(n_episodes=10)
            print(f"Movement Success Rate: {movement_results['success_rate']*100:.1f}%")

        if not args.target_only and not args.movement_only:
            combined_results = trainer.evaluate_combined(n_episodes=10)
            print(f"Combined Full Success: {combined_results['full_success']*100:.1f}%")

    trainer.close()


if __name__ == "__main__":
    main()
