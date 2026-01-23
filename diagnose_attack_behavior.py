"""Diagnose why trained model doesn't attack when in range."""

from pathlib import Path
from stable_baselines3 import PPO
from src.combat import CombatEnv
from src.world import WorldConfig, Action
from src.world.entities import CombatStyle

# Find model
model_paths = [
    Path("models/combat/best/best_model.zip"),
    Path("models/combat/combat_balanced_final.zip"),
]
model_path = next((p for p in model_paths if p.exists()), None)
if not model_path:
    print("No model found!")
    exit(1)

print(f"Loading: {model_path}\n")
model = PPO.load(str(model_path))

# Create env with close spawn (guaranteed in range)
config = WorldConfig(size=16, num_objects=0)
env = CombatEnv(
    world_config=config,
    num_enemies=1,
    combat_style=CombatStyle.BALANCED,
    max_steps=100,
    seed=42,
    close_spawn=True,
)

print("="*60)
print("DIAGNOSIS: What does the model do when in attack range?")
print("="*60)

# Run multiple episodes and track behavior
in_range_actions = []
in_range_can_attack_actions = []
in_range_on_cooldown_actions = []

for ep in range(10):
    obs, _ = env.reset(seed=100+ep)

    for step in range(100):
        npc = env.world.npc
        enemy = env.world.enemies[0]
        dist = npc.position.distance_to(enemy.position)
        in_range = dist <= npc.attack_range
        can_attack = npc.can_attack

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        action_name = Action(action).name

        if in_range:
            in_range_actions.append(action_name)
            if can_attack:
                in_range_can_attack_actions.append(action_name)
            else:
                in_range_on_cooldown_actions.append(action_name)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

env.close()

# Analyze
print("\n--- When IN RANGE (dist <= attack_range) ---")
if in_range_actions:
    from collections import Counter
    counts = Counter(in_range_actions)
    total = len(in_range_actions)
    print(f"Total observations in range: {total}")
    for action, count in counts.most_common():
        print(f"  {action:12}: {count:4} ({100*count/total:5.1f}%)")

print("\n--- When IN RANGE and CAN_ATTACK (ready to attack) ---")
if in_range_can_attack_actions:
    counts = Counter(in_range_can_attack_actions)
    total = len(in_range_can_attack_actions)
    print(f"Total observations ready to attack: {total}")
    for action, count in counts.most_common():
        print(f"  {action:12}: {count:4} ({100*count/total:5.1f}%)")

    attack_rate = counts.get('ATTACK', 0) / total * 100
    print(f"\n  >>> ATTACK RATE WHEN READY: {attack_rate:.1f}%")
    if attack_rate < 50:
        print("  >>> PROBLEM: Model doesn't attack when it should!")
else:
    print("  No observations where model was ready to attack")

print("\n--- When IN RANGE but ON COOLDOWN ---")
if in_range_on_cooldown_actions:
    counts = Counter(in_range_on_cooldown_actions)
    total = len(in_range_on_cooldown_actions)
    print(f"Total observations on cooldown: {total}")
    for action, count in counts.most_common():
        print(f"  {action:12}: {count:4} ({100*count/total:5.1f}%)")
else:
    print("  No observations while on cooldown")

# Also check observation values when ready to attack
print("\n--- Sample observation when ready to attack ---")
obs, _ = env.reset(seed=42)
env = CombatEnv(
    world_config=config,
    num_enemies=1,
    combat_style=CombatStyle.BALANCED,
    max_steps=100,
    seed=42,
    close_spawn=True,
)
obs, _ = env.reset()
npc = env.world.npc
enemy = env.world.enemies[0]
dist = npc.position.distance_to(enemy.position)

print(f"NPC position: ({npc.position.x:.3f}, {npc.position.y:.3f})")
print(f"Enemy position: ({enemy.position.x:.3f}, {enemy.position.y:.3f})")
print(f"Distance: {dist:.3f}, Attack range: {npc.attack_range}")
print(f"Can attack: {npc.can_attack}")
print(f"\nObservation (29 dims):")
print(f"  [0-1] NPC pos:      {obs[0]:.3f}, {obs[1]:.3f}")
print(f"  [2] Health ratio:   {obs[2]:.3f}")
print(f"  [3] Can attack:     {obs[3]:.3f}")  # Should be 1.0 if ready
print(f"  [4] Cooldown ratio: {obs[4]:.3f}")
print(f"  [5] Num enemies:    {obs[5]:.3f}")
print(f"  [6-7] Enemy1 rel:   {obs[6]:.3f}, {obs[7]:.3f}")
print(f"  [8] Enemy1 dist:    {obs[8]:.3f}")  # Should be small
print(f"  [9] Enemy1 HP:      {obs[9]:.3f}")
print(f"  [10] Enemy1 aggro:  {obs[10]:.3f}")

action, _ = model.predict(obs, deterministic=True)
print(f"\nModel predicts: {Action(int(action)).name}")

env.close()
