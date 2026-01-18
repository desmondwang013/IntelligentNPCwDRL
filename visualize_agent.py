"""
Visualize trained agent behavior.
Shows NPC movement, target location, and decision-making in real-time.
"""
import argparse
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from stable_baselines3 import PPO

from src.training import NPCEnv, SimpleNPCEnv
from src.world.world import Action


def print_world(env: Union[NPCEnv, SimpleNPCEnv], action: Optional[int], reward: float, step: int,
                total_reward: float, info: dict):
    """Print ASCII visualization of the world state."""
    runtime = env.runtime
    if runtime is None:
        return

    state = runtime.get_state()
    world_state = state["world"]
    world_size = world_state["world_size"]
    npc_pos = world_state["npc"]["position"]
    objects = world_state["objects"]

    # Get target info
    target_id = env._current_target_id
    target_obj = None
    for obj in objects:
        if obj["entity_id"] == target_id:
            target_obj = obj
            break

    # Calculate distance to target
    if target_obj:
        target_pos = target_obj["position"]
        distance = np.sqrt(
            (npc_pos["x"] - target_pos["x"])**2 +
            (npc_pos["y"] - target_pos["y"])**2
        )
    else:
        distance = -1
        target_pos = {"x": 0, "y": 0}

    # Clear screen (simple approach)
    print("\n" * 2)
    print("=" * 60)
    print(f"Step {step:3d} | Total Reward: {total_reward:7.2f} | Step Reward: {reward:6.2f}")
    print("=" * 60)

    # Action taken
    action_name = Action(action).name if action is not None else "NONE"
    print(f"Action: {action_name}")
    print(f"NPC Position:    ({npc_pos['x']:5.1f}, {npc_pos['y']:5.1f})")
    print(f"Target Position: ({target_pos['x']:5.1f}, {target_pos['y']:5.1f})")
    print(f"Distance:        {distance:5.2f} (threshold: {env.distance_threshold})")

    # Direction hint
    if target_obj:
        dx = target_pos["x"] - npc_pos["x"]
        dy = target_pos["y"] - npc_pos["y"]

        direction = []
        if abs(dy) > 0.5:
            direction.append("UP" if dy > 0 else "DOWN")
        if abs(dx) > 0.5:
            direction.append("RIGHT" if dx > 0 else "LEFT")

        optimal = " + ".join(direction) if direction else "AT TARGET"
        print(f"Optimal move:    {optimal}")

    # Simple ASCII grid (scaled down for visibility)
    grid_size = min(20, world_size)
    scale = world_size / grid_size

    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    # Place objects
    for obj in objects:
        gx = int(obj["position"]["x"] / scale)
        gy = int(obj["position"]["y"] / scale)
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        if obj["entity_id"] == target_id:
            grid[grid_size - 1 - gy][gx] = "T"  # Target
        else:
            grid[grid_size - 1 - gy][gx] = "o"  # Other object

    # Place NPC (overwrites if on same cell)
    npc_gx = int(npc_pos["x"] / scale)
    npc_gy = int(npc_pos["y"] / scale)
    npc_gx = max(0, min(grid_size - 1, npc_gx))
    npc_gy = max(0, min(grid_size - 1, npc_gy))
    grid[grid_size - 1 - npc_gy][npc_gx] = "N"

    print("\nWorld View (N=NPC, T=Target, o=object):")
    print("+" + "-" * grid_size + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * grid_size + "+")

    # Intent info
    intent = state.get("intent", {})
    if intent.get("has_intent"):
        print(f"\nIntent: {intent['intent']['text']}")


def run_episode(env: NPCEnv, model: Optional[PPO], delay: float = 0.1,
                use_greedy: bool = False) -> dict:
    """Run a single episode with visualization."""
    obs, info = env.reset()

    total_reward = 0
    step = 0
    done = False

    # Show initial state
    print_world(env, None, 0, step, total_reward, info)
    time.sleep(delay)

    while not done:
        # Get action
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif use_greedy:
            # Simple greedy: move toward target
            action = get_greedy_action(env)
        else:
            # Random action
            action = env.action_space.sample()

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1

        # Visualize
        print_world(env, action, reward, step, total_reward, info)
        time.sleep(delay)

    # Final summary
    print("\n" + "=" * 60)
    print("EPISODE COMPLETE")
    print("=" * 60)
    termination = info.get("termination_reason", "MAX_STEPS" if step >= 500 else "UNKNOWN")
    print(f"Termination: {termination}")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Success: {termination == 'INTENT_COMPLETED'}")

    return {
        "steps": step,
        "reward": total_reward,
        "success": termination == "INTENT_COMPLETED",
        "termination": termination,
    }


def get_greedy_action(env: NPCEnv) -> int:
    """Get greedy action that moves toward target."""
    runtime = env.runtime
    if runtime is None:
        return 4  # WAIT

    state = runtime.get_state()
    world_state = state["world"]
    npc_pos = world_state["npc"]["position"]

    # Find target
    target_id = env._current_target_id
    for obj in world_state["objects"]:
        if obj["entity_id"] == target_id:
            target_pos = obj["position"]
            break
    else:
        return 4  # WAIT if no target

    dx = target_pos["x"] - npc_pos["x"]
    dy = target_pos["y"] - npc_pos["y"]

    # Prioritize larger delta
    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 2  # RIGHT or LEFT
    elif abs(dy) > 0.1:
        return 0 if dy > 0 else 1  # UP or DOWN
    else:
        return 4  # WAIT (at target)


def main():
    parser = argparse.ArgumentParser(description="Visualize agent behavior")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (default: latest)")
    parser.add_argument("--world-size", type=int, default=8,
                        help="World size (default: 8)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Delay between steps in seconds")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy policy instead of trained model")
    parser.add_argument("--random", action="store_true",
                        help="Use random policy")
    parser.add_argument("--distance-threshold", type=float, default=2.0,
                        help="Success distance threshold")
    parser.add_argument("--simple", action="store_true",
                        help="Use SimpleNPCEnv (24-dim obs, no embeddings)")

    args = parser.parse_args()

    # Load model if specified
    model = None
    if not args.random and not args.greedy:
        model_path = args.model
        if model_path is None:
            # Try to find latest model (prefer simple if --simple flag)
            if args.simple:
                candidates = [
                    "models/simple/best/best_model.zip",
                    "models/simple/final_model.zip",
                ]
            else:
                candidates = [
                    "models/curriculum_v2/best/best_model.zip",
                    "models/curriculum_v2/final_model.zip",
                    "models/curriculum/best/best_model.zip",
                ]
            for path in candidates:
                if Path(path).exists():
                    model_path = path
                    break

        if model_path and Path(model_path).exists():
            print(f"Loading model from: {model_path}")
            model = PPO.load(model_path)
        else:
            print("No model found, using random policy")

    # Create environment
    if args.simple:
        print("Using SimpleNPCEnv (24-dim observation, no embeddings)")
        env = SimpleNPCEnv(
            world_size=args.world_size,
            max_steps_per_episode=500,
            distance_threshold=args.distance_threshold,
            seed=42,
        )
    else:
        env = NPCEnv(
            world_size=args.world_size,
            max_steps_per_episode=500,
            distance_threshold=args.distance_threshold,
            seed=42,
        )

    print(f"\nRunning {args.episodes} episodes")
    print(f"World size: {args.world_size}x{args.world_size}")
    print(f"Observation dim: {env.observation_space.shape[0]}")
    print(f"Policy: {'Greedy' if args.greedy else 'Model' if model else 'Random'}")
    print(f"Distance threshold: {args.distance_threshold}")
    print("\nPress Ctrl+C to stop\n")

    results = []
    try:
        for ep in range(args.episodes):
            print(f"\n{'#' * 60}")
            print(f"# EPISODE {ep + 1}/{args.episodes}")
            print(f"{'#' * 60}")

            result = run_episode(
                env, model,
                delay=args.delay,
                use_greedy=args.greedy
            )
            results.append(result)

            input("\nPress Enter for next episode...")

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        env.close()

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        successes = sum(1 for r in results if r["success"])
        print(f"Episodes: {len(results)}")
        print(f"Successes: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
        print(f"Avg Steps: {np.mean([r['steps'] for r in results]):.1f}")
        print(f"Avg Reward: {np.mean([r['reward'] for r in results]):.2f}")


if __name__ == "__main__":
    main()
