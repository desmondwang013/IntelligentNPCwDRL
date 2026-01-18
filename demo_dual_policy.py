"""
LEGACY DEMO: Two-Policy Architecture (learned target selection + movement).

In the current architecture:
- Target resolution is DETERMINISTIC (not learned) via Validator/Resolver
- RL only handles motor execution with structured goals (SimpleNPCEnv)

This demo showed the experimental learned target selection approach.
For the current architecture, use train_simple.py and visualize_agent.py.

Original description:
This demonstrates the learned target selection + movement architecture:
1. Target Selection Policy: instruction embedding → object index
2. Movement Policy: navigate to selected object

This is a short training run to verify everything works.
"""
from src.training import (
    TargetSelectionEnv,
    MovementEnv,
    DualPolicyTrainer,
    DualTrainerConfig,
)


def test_target_selection_env():
    """Test the target selection environment."""
    print("\n" + "="*60)
    print("Testing Target Selection Environment")
    print("="*60)

    env = TargetSelectionEnv()

    # Run a few episodes
    for episode in range(3):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}:")
        print(f"  Instruction: {info['instruction']}")
        print(f"  Correct answer: index {info['correct_object_index']}")

        # Random action (agent would learn to pick correctly)
        import random
        action = random.randint(0, 9)
        obs, reward, terminated, truncated, step_info = env.step(action)

        print(f"  Agent picked: index {action}")
        print(f"  Reward: {reward}")
        print(f"  Correct: {step_info['is_correct']}")

    env.close()
    print("\nTarget selection env: OK")


def test_movement_env():
    """Test the movement environment."""
    print("\n" + "="*60)
    print("Testing Movement Environment")
    print("="*60)

    env = MovementEnv()

    # Run one episode with random target
    obs, info = env.reset(options={"target_index": 3})
    print(f"Target index: {info['target_index']}")
    print(f"Target ID: {info['target_id']}")
    print(f"Instruction: {info['instruction']}")

    # Take a few random steps
    total_reward = 0
    for step in range(10):
        import random
        action = random.randint(0, 4)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"\nEpisode ended at step {step + 1}")
            print(f"  Reason: {step_info.get('termination_reason', 'max_steps')}")
            break

    print(f"Total reward (10 steps): {total_reward:.3f}")

    env.close()
    print("\nMovement env: OK")


def train_short_demo():
    """Run a short training demo."""
    print("\n" + "="*60)
    print("Training Demo (Short Run)")
    print("="*60)

    # Configure for quick test
    config = DualTrainerConfig(
        target_timesteps=2000,     # Very short for demo
        target_n_envs=2,
        movement_timesteps=3000,   # Very short for demo
        movement_n_envs=2,
        checkpoint_freq=5000,      # Won't checkpoint in this short run
    )

    trainer = DualPolicyTrainer(config)

    # Train both policies
    print("\n--- Training Target Selection Policy ---")
    trainer.train_target_selection(progress_bar=False)

    print("\n--- Training Movement Policy ---")
    trainer.train_movement(progress_bar=False)

    # Save models
    trainer.save("demo_dual_policy")

    # Evaluate
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)

    print("\nTarget Selection:")
    target_results = trainer.evaluate_target_selection(n_episodes=50)
    print(f"  Accuracy: {target_results['accuracy']*100:.1f}%")
    print(f"  ({target_results['correct']}/{target_results['total']} correct)")

    print("\nMovement:")
    movement_results = trainer.evaluate_movement(n_episodes=10)
    print(f"  Success rate: {movement_results['success_rate']*100:.1f}%")
    print(f"  Avg steps: {movement_results['avg_steps']:.1f}")

    print("\nCombined (Target Selection → Movement):")
    combined_results = trainer.evaluate_combined(n_episodes=10)
    print(f"  Target accuracy: {combined_results['target_accuracy']*100:.1f}%")
    print(f"  Navigation success: {combined_results['navigation_success']*100:.1f}%")
    print(f"  Full success: {combined_results['full_success']*100:.1f}%")

    trainer.close()


def main():
    print("="*60)
    print("DUAL POLICY ARCHITECTURE DEMO")
    print("Phase 1: Learned Target Selection + Movement")
    print("="*60)

    # Test environments
    test_target_selection_env()
    test_movement_env()

    # Train
    train_short_demo()

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nFor full training, use:")
    print("  python train_dual_policy.py --target-steps 50000 --movement-steps 100000")


if __name__ == "__main__":
    main()
