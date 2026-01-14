"""
Demo script to validate the training pipeline works.
Does a very short training run to test everything connects.

Run from project root: python demo_training.py
"""
from src.training import NPCEnv, Trainer, TrainerConfig
from src.runtime import RuntimeConfig
from src.reward import RewardConfig


def test_environment():
    """Test the Gym environment wrapper."""
    print("=== Testing NPCEnv ===\n")

    env = NPCEnv(seed=42)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print()

    # Test reset
    obs, info = env.reset()
    print(f"Reset - obs shape: {obs.shape}")
    print(f"Target: {info.get('target_id')}")
    print(f"Instruction: {info.get('instruction')}")
    print()

    # Test a few steps
    print("Running 10 steps with random actions...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            reason = info.get("termination_reason", "max_steps")
            print(f"  Episode ended at step {i+1}: {reason}")
            break

    print(f"Total reward: {total_reward:.2f}")
    print()

    env.close()
    print("Environment test passed!\n")


def test_short_training():
    """Test a very short training run."""
    print("=== Testing Short Training ===\n")

    # Very short config for testing
    config = TrainerConfig(
        total_timesteps=1000,      # Very short
        n_envs=2,                  # Fewer envs
        n_steps=128,               # Smaller buffer
        batch_size=32,
        save_freq=500,
        eval_freq=500,
        n_eval_episodes=2,
        save_dir="models_test",
        log_dir="logs_test",
    )

    trainer = Trainer(config=config)
    trainer.setup()

    print(f"Model parameters: {trainer._count_parameters()}")
    print()

    # Train
    print("Training for 1000 timesteps...")
    stats = trainer.train(progress_bar=False)

    # Quick eval
    print("\nEvaluating...")
    results = trainer.evaluate(n_episodes=3)
    print(f"Mean reward: {results['mean_reward']:.2f}")
    print(f"Mean length: {results['mean_length']:.1f}")

    # Test save/load
    print("\nTesting save/load...")
    trainer.save("test_model")
    trainer.load("models_test/test_model")
    print("Save/load works!")

    # Test prediction
    env = NPCEnv(seed=999)
    obs, _ = env.reset()
    action = trainer.predict(obs)
    print(f"Prediction test - action: {action}")

    trainer.close()
    env.close()

    print("\nShort training test passed!")


def main():
    print("=" * 50)
    print("Training Pipeline Demo")
    print("=" * 50)
    print()

    try:
        test_environment()
        test_short_training()
        print("\n" + "=" * 50)
        print("All tests passed! Training pipeline is ready.")
        print("=" * 50)
        print("\nTo run full training:")
        print("  python train.py --timesteps 100000")
        print("\nTo monitor with tensorboard:")
        print("  tensorboard --logdir logs")

    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("\nInstall with: pip install -r requirements.txt")

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
