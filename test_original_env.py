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
    """Train for a short time and check success rate."""
    print("="*60)
    print("Testing short training with original NPCEnv...")
    print("="*60)

    config = TrainerConfig(
        total_timesteps=10000,  # Very short
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

    print(f"Success rate: {results['success_rate']*100:.1f}%")
    print(f"Avg reward: {results['avg_reward']:.2f}")
    print(f"Avg steps: {results['avg_steps']:.1f}")

    trainer.close()

    return results['success_rate']


if __name__ == "__main__":
    test_env_basics()
    success_rate = test_short_training()

    print("\n" + "="*60)
    if success_rate > 0:
        print(f"BASELINE WORKS: {success_rate*100:.1f}% success rate")
        print("Original NPCEnv is functional.")
    else:
        print("BASELINE BROKEN: 0% success rate")
        print("Something fundamental is wrong.")
    print("="*60)
