"""
Test the original NPCEnv to confirm it still works.
This is our baseline - if this fails, something fundamental broke.
"""
from src.training import NPCEnv, Trainer, TrainerConfig


def test_env_basics():
    """Test that the environment runs at all."""
    print("="*60)
    print("Testing NPCEnv basics...")
    print("="*60)

    env = NPCEnv()
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Target ID: {info['target_id']}")
    print(f"Instruction: {info['instruction']}")

    # Take 10 random steps
    import random
    total_reward = 0
    for i in range(10):
        action = random.randint(0, 4)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {i+1}: {step_info.get('termination_reason', 'truncated')}")
            break

    print(f"Total reward (10 steps): {total_reward:.3f}")
    env.close()
    print("Basic test: PASSED\n")


def test_short_training():
    """Train for a short time and check if learning happens."""
    print("="*60)
    print("Testing short training with original NPCEnv...")
    print("="*60)

    config = TrainerConfig(
        total_timesteps=30000,  # Slightly longer for meaningful test
        n_envs=2,
        save_dir="models/test_original",
        log_dir="logs/test_original",
    )

    trainer = Trainer(config)
    trainer.setup()
    trainer.train(progress_bar=False)

    # Evaluate
    print("\nEvaluating...")
    results = trainer.evaluate(n_episodes=20)

    print(f"Mean reward: {results['mean_reward']:.2f}")
    print(f"Mean episode length: {results['mean_length']:.1f}")

    trainer.close()

    # Success = positive mean reward OR episodes ending before timeout (500)
    # If mean_length < 400, agent is completing some tasks
    success = results['mean_length'] < 400 or results['mean_reward'] > 0
    return success, results


if __name__ == "__main__":
    test_env_basics()
    success, results = test_short_training()

    print("\n" + "="*60)
    if success:
        print("BASELINE WORKS")
        print(f"  Mean reward: {results['mean_reward']:.2f}")
        print(f"  Mean length: {results['mean_length']:.1f}")
        print("Original NPCEnv is functional.")
    else:
        print("BASELINE SHOWS NO LEARNING YET")
        print(f"  Mean reward: {results['mean_reward']:.2f}")
        print(f"  Mean length: {results['mean_length']:.1f} (timeout is 500)")
        print("May need more training steps.")
    print("="*60)
