"""Debug combat training to see what's happening."""

import numpy as np
from src.combat import CombatEnv, CombatRewardConfig
from src.world import WorldConfig, Action
from src.world.entities import CombatStyle


def debug_random_agent():
    """Run random agent and see rewards."""
    print("=== Random Agent Debug ===\n")

    config = WorldConfig(size=16, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        seed=42,
        close_spawn=True,
    )

    obs, info = env.reset()
    print(f"Initial state:")
    print(f"  NPC pos: ({env.world.npc.position.x:.2f}, {env.world.npc.position.y:.2f})")
    print(f"  Enemy pos: ({env.world.enemies[0].position.x:.2f}, {env.world.enemies[0].position.y:.2f})")
    print(f"  Distance: {env.world.npc.position.distance_to(env.world.enemies[0].position):.3f}")
    print(f"  Attack range: {env.world.npc.attack_range}")
    print()

    total_reward = 0
    action_counts = {a.name: 0 for a in Action}
    reward_by_action = {a.name: [] for a in Action}

    for step in range(100):
        action = env.action_space.sample()
        action_name = Action(action).name
        action_counts[action_name] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        reward_by_action[action_name].append(reward)

        if step < 20:  # Show first 20 steps
            breakdown = {k: v for k, v in info['reward_breakdown'].items() if v != 0 and k != 'total'}
            print(f"Step {step+1}: {action_name:12} -> reward={reward:+.3f} {breakdown}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            print(f"  NPC alive: {info['npc_alive']}, Enemies alive: {info['enemies_alive']}")
            break

    print(f"\n--- Summary ---")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Action distribution: {action_counts}")
    print(f"\nAverage reward by action:")
    for action_name, rewards in reward_by_action.items():
        if rewards:
            print(f"  {action_name:12}: avg={np.mean(rewards):+.3f}, count={len(rewards)}")

    env.close()


def debug_always_attack():
    """Run agent that always attacks."""
    print("\n\n=== Always Attack Agent ===\n")

    config = WorldConfig(size=16, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        seed=42,
        close_spawn=True,
    )

    obs, info = env.reset()
    total_reward = 0

    for step in range(100):
        action = Action.ATTACK.value  # Always attack

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        breakdown = {k: v for k, v in info['reward_breakdown'].items() if v != 0 and k != 'total'}
        print(f"Step {step+1}: ATTACK -> reward={reward:+.3f} {breakdown}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            print(f"  NPC alive: {info['npc_alive']}, Enemies alive: {info['enemies_alive']}")
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()


def debug_observation():
    """Check what the observation looks like."""
    print("\n\n=== Observation Debug ===\n")

    config = WorldConfig(size=16, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        seed=42,
        close_spawn=True,
    )

    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"\nObservation breakdown:")

    labels = env.obs_builder.get_observation_labels()
    for i, (label, val) in enumerate(zip(labels, obs)):
        print(f"  [{i:2d}] {label:20}: {val:+.4f}")

    env.close()


def debug_ppo_actions():
    """Load a PPO model and see what actions it takes."""
    print("\n\n=== PPO Model Debug ===\n")

    try:
        from stable_baselines3 import PPO
        from pathlib import Path

        model_path = Path("models/combat/best/best_model.zip")
        if not model_path.exists():
            model_path = Path("models/combat/combat_balanced_final.zip")

        if not model_path.exists():
            print("No trained model found. Skipping PPO debug.")
            return

        print(f"Loading model from {model_path}")
        model = PPO.load(str(model_path))

        config = WorldConfig(size=16, num_objects=0)
        env = CombatEnv(
            world_config=config,
            num_enemies=1,
            combat_style=CombatStyle.BALANCED,
            max_steps=100,
            seed=42,
            close_spawn=True,
        )

        obs, info = env.reset()
        action_counts = {a.name: 0 for a in Action}

        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            action_name = Action(action).name
            action_counts[action_name] += 1

            obs, reward, terminated, truncated, info = env.step(action)

            if step < 10:
                print(f"Step {step+1}: {action_name}")

            if terminated or truncated:
                break

        print(f"\nAction distribution: {action_counts}")
        env.close()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    debug_random_agent()
    debug_always_attack()
    debug_observation()
    debug_ppo_actions()
