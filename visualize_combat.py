"""
Visualize trained combat agent behavior.
Shows NPC vs enemies with health bars, actions, and combat stats.
"""
import argparse
import time
import os
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from src.combat import CombatEnv, CombatRewardConfig
from src.world import WorldConfig, Action
from src.world.entities import CombatStyle


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def health_bar(current: int, maximum: int, width: int = 10) -> str:
    """Create ASCII health bar."""
    if maximum == 0:
        return "[" + "?" * width + "]"
    filled = int((current / maximum) * width)
    empty = width - filled
    return "[" + "#" * filled + "-" * empty + "]"


def print_combat_state(env: CombatEnv, action: Optional[int], reward: float,
                       step: int, total_reward: float, info: dict):
    """Print ASCII visualization of combat state."""
    world = env.world
    npc = world.npc
    enemies = world.enemies
    world_size = world.config.size

    clear_screen()

    # Header
    print("=" * 60)
    print(f"  COMBAT VISUALIZATION - Step {step}")
    print("=" * 60)

    # Action info
    action_name = Action(action).name if action is not None else "---"
    print(f"\n  Action: {action_name:12}  Reward: {reward:+.2f}  Total: {total_reward:.2f}")

    # Reward breakdown (non-zero only)
    if 'reward_breakdown' in info:
        breakdown = {k: f"{v:+.2f}" for k, v in info['reward_breakdown'].items()
                     if v != 0 and k != 'total'}
        if breakdown:
            print(f"  Breakdown: {breakdown}")

    # NPC status
    print("\n" + "-" * 60)
    print(f"  NPC {health_bar(npc.health, npc.max_health, 15)} {npc.health:2d}/{npc.max_health} HP")
    print(f"      Position: ({npc.position.x:5.1f}, {npc.position.y:5.1f})")
    cooldown_str = "READY" if npc.can_attack else f"CD: {npc.current_cooldown}"
    print(f"      Attack: {cooldown_str}  |  Style: {npc.combat_style.name}")

    # Enemies status
    print("\n  ENEMIES:")
    for i, enemy in enumerate(enemies):
        if enemy.is_alive:
            dist = npc.position.distance_to(enemy.position)
            in_range = "IN RANGE" if dist <= npc.attack_range else f"dist: {dist:.1f}"
            aggro = "AGGRO!" if enemy.is_aggro else "passive"
            print(f"    [{i}] {health_bar(enemy.health, enemy.max_health, 10)} "
                  f"{enemy.health}/{enemy.max_health} HP  |  {in_range}  |  {aggro}")
        else:
            print(f"    [{i}] [  DEAD  ]")

    print("-" * 60)

    # ASCII Grid visualization
    grid_size = 24
    scale = world_size / grid_size

    # Initialize grid
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    # NPC grid position
    npc_gx = int(npc.position.x / scale)
    npc_gy = int(npc.position.y / scale)
    npc_gx = max(0, min(grid_size - 1, npc_gx))
    npc_gy = max(0, min(grid_size - 1, npc_gy))
    npc_row = grid_size - 1 - npc_gy

    # Draw attack range first (larger radius for visibility)
    attack_range_grid = max(2, int(npc.attack_range / scale) + 1)
    for dy in range(-attack_range_grid, attack_range_grid + 1):
        for dx in range(-attack_range_grid, attack_range_grid + 1):
            gy = npc_row + dy
            gx = npc_gx + dx
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                actual_dist = ((dx * scale) ** 2 + (dy * scale) ** 2) ** 0.5
                if actual_dist <= npc.attack_range * 1.5:  # Slightly expanded for visibility
                    grid[gy][gx] = "+"

    # Place enemies (on top of attack range)
    for i, enemy in enumerate(enemies):
        if enemy.is_alive:
            gx = int(enemy.position.x / scale)
            gy = int(enemy.position.y / scale)
            gx = max(0, min(grid_size - 1, gx))
            gy = max(0, min(grid_size - 1, gy))
            grid[grid_size - 1 - gy][gx] = "E"
        else:
            # Show dead enemies with x
            gx = int(enemy.position.x / scale)
            gy = int(enemy.position.y / scale)
            gx = max(0, min(grid_size - 1, gx))
            gy = max(0, min(grid_size - 1, gy))
            grid[grid_size - 1 - gy][gx] = "x"

    # Place NPC (use X if on enemy position)
    if grid[npc_row][npc_gx] == "E":
        grid[npc_row][npc_gx] = "X"  # Combat!
    else:
        grid[npc_row][npc_gx] = "@"

    # Print grid with border
    print("\n  " + "+" + "-" * grid_size + "+")
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  " + "+" + "-" * grid_size + "+")
    print("  @ = NPC  E = Enemy  X = Combat!  + = Range  x = Dead")

    # Combat stats
    print("\n" + "=" * 60)


def run_combat_episode(env: CombatEnv, model: Optional[PPO], delay: float = 0.3,
                       deterministic: bool = True) -> dict:
    """Run a single combat episode with visualization."""
    obs, info = env.reset()

    total_reward = 0
    step = 0
    done = False

    damage_dealt = 0
    damage_taken = 0
    kills = 0

    # Show initial state
    print_combat_state(env, None, 0, step, total_reward, info)
    time.sleep(delay)

    while not done:
        # Get action
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
        else:
            action = env.action_space.sample()

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1

        # Track combat stats
        breakdown = info.get('reward_breakdown', {})
        if 'damage_dealt' in breakdown:
            damage_dealt += env.world.last_damage_dealt
        if 'damage_taken' in breakdown:
            damage_taken += env.world.last_damage_taken
        if 'kill' in breakdown:
            kills += 1

        # Visualize
        print_combat_state(env, action, reward, step, total_reward, info)
        time.sleep(delay)

    # Final summary
    clear_screen()
    print("\n" + "=" * 60)
    print("  COMBAT COMPLETE!")
    print("=" * 60)

    npc_alive = env.world.npc.is_alive
    enemies_alive = len(env.world.get_alive_enemies())
    victory = npc_alive and enemies_alive == 0

    print(f"\n  Result: {'VICTORY!' if victory else 'DEFEAT' if not npc_alive else 'TIMEOUT'}")
    print(f"\n  Steps:        {step}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Damage Dealt: {damage_dealt}")
    print(f"  Damage Taken: {damage_taken}")
    print(f"  Kills:        {kills}")
    print(f"  NPC Health:   {env.world.npc.health}/{env.world.npc.max_health}")
    print("\n" + "=" * 60)

    return {
        "steps": step,
        "reward": total_reward,
        "victory": victory,
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
        "kills": kills,
        "npc_health": env.world.npc.health,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize combat agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model")
    parser.add_argument("--style", type=str, default="balanced",
                        choices=["aggressive", "balanced", "passive"],
                        help="Combat style")
    parser.add_argument("--enemies", type=int, default=1,
                        help="Number of enemies")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Delay between steps (seconds)")
    parser.add_argument("--world-size", type=int, default=16,
                        help="World size")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions instead of model")
    parser.add_argument("--no-close-spawn", action="store_true",
                        help="Disable close spawn (enemies spawn far)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic (non-deterministic) actions")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for user input between episodes")

    args = parser.parse_args()

    # Parse combat style
    style_map = {
        "aggressive": CombatStyle.AGGRESSIVE,
        "balanced": CombatStyle.BALANCED,
        "passive": CombatStyle.PASSIVE,
    }
    combat_style = style_map[args.style]

    # Load model
    model = None
    if not args.random:
        model_path = args.model
        if model_path is None:
            # Try to find model
            candidates = [
                f"models/combat/combat_{args.style}_final.zip",
                "models/combat/best/best_model.zip",
                "models/combat/combat_balanced_final.zip",
            ]
            for path in candidates:
                if Path(path).exists():
                    model_path = path
                    break

        if model_path and Path(model_path).exists():
            print(f"Loading model from: {model_path}")
            model = PPO.load(model_path)
        else:
            print("No model found. Use --random for random actions or --model to specify path.")
            return

    # Create environment
    config = WorldConfig(size=args.world_size, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=args.enemies,
        combat_style=combat_style,
        max_steps=200,
        seed=42,
        close_spawn=not args.no_close_spawn,
    )

    print(f"\nCombat Visualization")
    print(f"  Style: {args.style}")
    print(f"  Enemies: {args.enemies}")
    print(f"  World Size: {args.world_size}")
    print(f"  Close Spawn: {not args.no_close_spawn}")
    print(f"  Policy: {'Random' if args.random else 'Model'}")
    print(f"\nPress Ctrl+C to stop\n")

    if not args.no_wait:
        input("Press Enter to start...")

    results = []
    try:
        for ep in range(args.episodes):
            # Reset with different seed each episode
            env._seed = 42 + ep

            result = run_combat_episode(
                env, model,
                delay=args.delay,
                deterministic=not args.stochastic
            )
            results.append(result)

            if ep < args.episodes - 1 and not args.no_wait:
                input("\nPress Enter for next episode...")

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        env.close()

    # Session summary
    if results:
        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        victories = sum(1 for r in results if r["victory"])
        print(f"  Episodes:     {len(results)}")
        print(f"  Victories:    {victories}/{len(results)} ({100*victories/len(results):.0f}%)")
        print(f"  Avg Steps:    {np.mean([r['steps'] for r in results]):.1f}")
        print(f"  Avg Reward:   {np.mean([r['reward'] for r in results]):.2f}")
        print(f"  Avg Damage:   {np.mean([r['damage_dealt'] for r in results]):.1f} dealt, "
              f"{np.mean([r['damage_taken'] for r in results]):.1f} taken")
        print("=" * 60)


if __name__ == "__main__":
    main()
