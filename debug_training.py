"""Debug what's happening during training."""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.combat import CombatEnv, CombatRewardConfig
from src.world import WorldConfig, Action
from src.world.entities import CombatStyle


def debug_training_setup():
    """Debug the training environment setup."""
    print("=== Training Setup Debug ===\n")

    # Replicate training setup
    world_config = WorldConfig(size=16, num_objects=0)
    reward_config = CombatRewardConfig(phase=1)

    raw_envs = []
    for i in range(2):
        env = CombatEnv(
            world_config=world_config,
            num_enemies=1,
            combat_style=CombatStyle.BALANCED,
            max_steps=500,
            reward_config=reward_config,
            seed=42 + i,
            close_spawn=True,
        )
        raw_envs.append(env)
        print(f"Env {i}: close_spawn={env.close_spawn}, phase={env.reward_calculator.config.phase}")

    # Wrap with Monitor
    monitored_envs = [Monitor(env) for env in raw_envs]

    # Create vectorized environment
    vec_env = DummyVecEnv([lambda e=env: e for env in monitored_envs])

    print(f"\nVecEnv observation space: {vec_env.observation_space}")
    print(f"VecEnv action space: {vec_env.action_space}")

    # Reset and check
    obs = vec_env.reset()
    print(f"\nAfter reset, checking underlying envs:")
    for i, env in enumerate(raw_envs):
        npc = env.world.npc
        enemy = env.world.enemies[0]
        dist = npc.position.distance_to(enemy.position)
        print(f"  Env {i}: NPC-Enemy dist={dist:.3f}, in_range={dist <= npc.attack_range}")

    # Run a few steps with random actions
    print("\n--- Running 20 random steps ---")
    total_rewards = [0, 0]
    for step in range(20):
        actions = [vec_env.action_space.sample() for _ in range(2)]
        obs, rewards, dones, infos = vec_env.step(actions)

        for i, (action, reward) in enumerate(zip(actions, rewards)):
            total_rewards[i] += reward
            if step < 5:
                print(f"  Step {step+1}, Env {i}: action={Action(action).name}, reward={reward:.3f}")

    print(f"\nTotal rewards after 20 steps: {total_rewards}")

    # Check if attack gives bonus
    print("\n--- Testing attack specifically ---")
    obs = vec_env.reset()
    for step in range(5):
        actions = [Action.ATTACK.value, Action.ATTACK.value]  # Both attack
        obs, rewards, dones, infos = vec_env.step(actions)
        print(f"  Step {step+1}: rewards={rewards}")

    vec_env.close()


def debug_ppo_initial():
    """Check what a fresh PPO does."""
    print("\n\n=== Fresh PPO Debug ===\n")

    world_config = WorldConfig(size=16, num_objects=0)
    reward_config = CombatRewardConfig(phase=1)

    env = CombatEnv(
        world_config=world_config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=500,
        reward_config=reward_config,
        seed=42,
        close_spawn=True,
    )
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    # Create fresh PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        verbose=0,
    )

    print("Action probabilities from fresh policy:")
    obs = vec_env.reset()

    # Get action probabilities
    import torch
    obs_tensor = torch.FloatTensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.numpy()[0]

    for i, prob in enumerate(probs):
        print(f"  {Action(i).name:12}: {prob:.3f}")

    # Run a short training
    print("\n--- Training for 1000 steps ---")
    model.learn(total_timesteps=1000, progress_bar=False)

    print("\nAction probabilities after 1000 steps:")
    obs = vec_env.reset()
    obs_tensor = torch.FloatTensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.numpy()[0]

    for i, prob in enumerate(probs):
        print(f"  {Action(i).name:12}: {prob:.3f}")

    # Test what actions it takes
    print("\n--- Actions taken by trained policy ---")
    action_counts = {a.name: 0 for a in Action}
    obs = vec_env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=False)
        action_counts[Action(action[0]).name] += 1
        obs, _, done, _ = vec_env.step(action)
        if done[0]:
            obs = vec_env.reset()

    print(f"Action distribution: {action_counts}")

    vec_env.close()


if __name__ == "__main__":
    debug_training_setup()
    debug_ppo_initial()
