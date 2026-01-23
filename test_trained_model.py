"""Test what a trained combat model actually does."""

import sys
from pathlib import Path

# Check for model
model_paths = [
    Path("models/combat/best/best_model.zip"),
    Path("models/combat/combat_balanced_final.zip"),
]

model_path = None
for p in model_paths:
    if p.exists():
        model_path = p
        break

if model_path is None:
    print("No trained model found!")
    print("Expected at: models/combat/best/best_model.zip")
    print("         or: models/combat/combat_balanced_final.zip")
    sys.exit(1)

print(f"Loading model from: {model_path}\n")

from stable_baselines3 import PPO
from src.combat import CombatEnv, CombatRewardConfig
from src.world import WorldConfig, Action
from src.world.entities import CombatStyle

# Load model
model = PPO.load(str(model_path))

# Create test env
config = WorldConfig(size=16, num_objects=0)
env = CombatEnv(
    world_config=config,
    num_enemies=1,
    combat_style=CombatStyle.BALANCED,
    max_steps=100,
    seed=42,
    close_spawn=True,
)

print("="*50)
print("Testing trained model behavior")
print("="*50)

# Run 3 episodes
for ep in range(3):
    print(f"\n--- Episode {ep+1} ---")
    obs, info = env.reset(seed=42+ep)

    npc = env.world.npc
    enemy = env.world.enemies[0]
    print(f"Initial: NPC at ({npc.position.x:.2f}, {npc.position.y:.2f}), "
          f"Enemy at ({enemy.position.x:.2f}, {enemy.position.y:.2f})")
    print(f"Distance: {npc.position.distance_to(enemy.position):.3f}")

    action_counts = {a.name: 0 for a in Action if a.value < 6}
    total_reward = 0

    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        action_name = Action(action).name
        action_counts[action_name] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step < 10:
            print(f"  Step {step+1}: {action_name:12} reward={reward:+.3f}")

        if terminated or truncated:
            print(f"  ... Episode ended at step {step+1}")
            break

    print(f"\nAction distribution: {action_counts}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Enemies alive: {info['enemies_alive']}, NPC alive: {info['npc_alive']}")

env.close()
