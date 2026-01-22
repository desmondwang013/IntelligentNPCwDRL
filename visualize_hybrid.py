"""
Visualize hybrid navigation: A* pathfinding + RL motor control.

Tests the navigation system on larger, more complex worlds with many obstacles.
Shows planned path, waypoints, and RL agent following them.
"""
import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.training import SimpleNPCEnv
from src.world.world import Action
from src.navigation import HybridNavigator, HybridNavigatorConfig, NavigationTarget
from src.observation.simple_builder import SimpleObservationBuilder, SimpleObservationConfig


def print_world_hybrid(
    env: SimpleNPCEnv,
    navigator: HybridNavigator,
    action: Optional[int],
    reward: float,
    step: int,
    total_reward: float,
    info: dict,
    current_waypoint: Optional[NavigationTarget],
):
    """Print ASCII visualization with path and waypoints."""
    runtime = env.runtime
    if runtime is None:
        return

    state = runtime.get_state()
    world_state = state["world"]
    world_size = world_state["world_size"]
    npc_pos = world_state["npc"]["position"]
    objects = world_state["objects"]

    # Get final target info
    target_id = env._current_target_id
    target_obj = None
    for obj in objects:
        if obj["entity_id"] == target_id:
            target_obj = obj
            break

    target_pos = target_obj["position"] if target_obj else {"x": 0, "y": 0}

    # Calculate distances
    dist_to_target = np.sqrt(
        (npc_pos["x"] - target_pos["x"])**2 +
        (npc_pos["y"] - target_pos["y"])**2
    ) if target_obj else -1

    dist_to_waypoint = -1
    if current_waypoint:
        dist_to_waypoint = np.sqrt(
            (npc_pos["x"] - current_waypoint.x)**2 +
            (npc_pos["y"] - current_waypoint.y)**2
        )

    # Navigation state
    nav_state = navigator.get_state()

    # Print header
    print("\n" * 2)
    print("=" * 70)
    print(f"Step {step:3d} | Reward: {total_reward:7.2f} | Waypoints: {nav_state.waypoints_remaining}/{nav_state.total_waypoints}")
    print("=" * 70)

    # Action info
    action_name = Action(action).name if action is not None else "NONE"
    blocked_str = "YES" if env._action_blocked else "no"
    print(f"Action: {action_name:12s} | Blocked: {blocked_str}")

    # Position info
    print(f"NPC Position:      ({npc_pos['x']:5.1f}, {npc_pos['y']:5.1f})")
    if current_waypoint:
        wp_type = "FINAL" if current_waypoint.is_final else "waypoint"
        print(f"Current Waypoint:  ({current_waypoint.x:5.1f}, {current_waypoint.y:5.1f}) [{wp_type}]")
        print(f"Dist to Waypoint:  {dist_to_waypoint:5.2f}")
    print(f"Final Target:      ({target_pos['x']:5.1f}, {target_pos['y']:5.1f})")
    print(f"Dist to Target:    {dist_to_target:5.2f} (threshold: {env.distance_threshold})")

    # Navigation status
    if nav_state.is_stuck:
        print("*** NAVIGATION STUCK - No valid path! ***")
    elif nav_state.is_complete:
        print("*** NAVIGATION COMPLETE ***")

    # Build ASCII grid
    grid_size = min(25, world_size)
    scale = world_size / grid_size

    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    # Place waypoints first (so objects/NPC can overwrite)
    waypoints = navigator.waypoints
    for i, (wx, wy) in enumerate(waypoints):
        gx = int(wx / scale)
        gy = int(wy / scale)
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        grid[grid_size - 1 - gy][gx] = "*"

    # Place objects
    for obj in objects:
        gx = int(obj["position"]["x"] / scale)
        gy = int(obj["position"]["y"] / scale)
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        if obj["entity_id"] == target_id:
            grid[grid_size - 1 - gy][gx] = "T"  # Target
        else:
            grid[grid_size - 1 - gy][gx] = "o"  # Obstacle

    # Place current waypoint marker
    if current_waypoint and not current_waypoint.is_final:
        gx = int(current_waypoint.x / scale)
        gy = int(current_waypoint.y / scale)
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        grid[grid_size - 1 - gy][gx] = "W"

    # Place NPC (overwrites everything)
    npc_gx = int(npc_pos["x"] / scale)
    npc_gy = int(npc_pos["y"] / scale)
    npc_gx = max(0, min(grid_size - 1, npc_gx))
    npc_gy = max(0, min(grid_size - 1, npc_gy))
    grid[grid_size - 1 - npc_gy][npc_gx] = "N"

    print(f"\nWorld View (N=NPC, T=Target, W=Waypoint, *=path, o=obstacle):")
    print("+" + "-" * grid_size + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * grid_size + "+")


def run_episode_hybrid(
    env: SimpleNPCEnv,
    model: Optional[PPO],
    navigator: HybridNavigator,
    obs_builder: SimpleObservationBuilder,
    delay: float = 0.1,
) -> dict:
    """Run episode with hybrid navigation."""
    # Reset environment
    obs, info = env.reset()

    # Get world state and positions
    state = env.runtime.get_state()
    world_state = state["world"]
    npc_pos = world_state["npc"]["position"]

    # Find target position
    target_id = env._current_target_id
    target_pos = None
    for obj in world_state["objects"]:
        if obj["entity_id"] == target_id:
            target_pos = obj["position"]
            break

    if target_pos is None:
        print("ERROR: Could not find target!")
        return {"steps": 0, "reward": 0, "success": False, "termination": "ERROR"}

    # Compute path using navigator
    path_found = navigator.set_goal(
        npc_pos["x"], npc_pos["y"],
        target_pos["x"], target_pos["y"],
        world_state
    )

    if not path_found:
        print("WARNING: No path found! Navigator will report stuck.")

    # Show initial planned path
    print("\n" + "=" * 70)
    print("PLANNED PATH")
    print("=" * 70)
    print(f"Start: ({npc_pos['x']:.1f}, {npc_pos['y']:.1f})")
    print(f"Goal:  ({target_pos['x']:.1f}, {target_pos['y']:.1f})")
    print(f"Waypoints: {len(navigator.waypoints)}")
    for i, (wx, wy) in enumerate(navigator.waypoints):
        marker = "(FINAL)" if i == len(navigator.waypoints) - 1 else ""
        print(f"  {i+1}. ({wx:.1f}, {wy:.1f}) {marker}")
    print("=" * 70)

    total_reward = 0
    step = 0
    done = False

    # Get initial waypoint
    current_waypoint = navigator.get_current_target()

    # Show initial state
    print_world_hybrid(env, navigator, None, 0, step, total_reward, info, current_waypoint)
    time.sleep(delay)

    while not done:
        # Get current waypoint for observation
        current_waypoint = navigator.get_current_target()

        if current_waypoint is None:
            # Navigation complete or stuck
            if not navigator.get_state().is_complete:
                print("Navigator has no waypoint but not complete - stuck?")
            break

        # Build observation with waypoint as target
        state = env.runtime.get_state()
        world_state = state["world"]
        npc_pos = world_state["npc"]["position"]
        intent_age = env.runtime.intent_manager.get_intent_age(env.runtime.tick)

        # Use waypoint position instead of final target
        obs = obs_builder.build(
            world_state=world_state,
            target_position=(current_waypoint.x, current_waypoint.y),
            intent_age_ticks=intent_age,
            distance_threshold=env.distance_threshold,
            previous_action=env._previous_action,
            action_blocked=env._action_blocked,
        )

        # Get action from model
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            # Fallback to greedy toward waypoint
            action = get_greedy_action_to_point(
                npc_pos["x"], npc_pos["y"],
                current_waypoint.x, current_waypoint.y
            )

        # Take step in environment
        _, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1

        # Update navigator with new position
        state = env.runtime.get_state()
        world_state = state["world"]
        npc_pos = world_state["npc"]["position"]
        navigator.update(npc_pos["x"], npc_pos["y"], world_state)

        # Get updated waypoint
        current_waypoint = navigator.get_current_target()

        # Visualize
        print_world_hybrid(env, navigator, action, reward, step, total_reward, info, current_waypoint)
        time.sleep(delay)

        # Check if navigator says we're done (reached via waypoints)
        nav_state = navigator.get_state()
        if nav_state.is_complete:
            print("\n*** Navigator reports: PATH COMPLETE ***")
            # Don't break - let environment confirm with its own threshold

    # Final summary
    print("\n" + "=" * 70)
    print("EPISODE COMPLETE")
    print("=" * 70)
    termination = info.get("termination_reason", "MAX_STEPS" if step >= 500 else "UNKNOWN")
    nav_state = navigator.get_state()
    print(f"Termination: {termination}")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Nav Complete: {nav_state.is_complete}")
    print(f"Nav Stuck: {nav_state.is_stuck}")
    print(f"Success: {termination == 'INTENT_COMPLETED'}")

    return {
        "steps": step,
        "reward": total_reward,
        "success": termination == "INTENT_COMPLETED",
        "termination": termination,
        "nav_complete": nav_state.is_complete,
        "nav_stuck": nav_state.is_stuck,
    }


def get_greedy_action_to_point(npc_x: float, npc_y: float, target_x: float, target_y: float) -> int:
    """Get greedy action moving toward a point."""
    dx = target_x - npc_x
    dy = target_y - npc_y

    if abs(dx) > abs(dy):
        return 3 if dx > 0 else 2  # RIGHT or LEFT
    elif abs(dy) > 0.1:
        return 0 if dy > 0 else 1  # UP or DOWN
    else:
        return 4  # WAIT


def main():
    parser = argparse.ArgumentParser(description="Test hybrid navigation (A* + RL)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model")
    parser.add_argument("--world-size", type=int, default=16,
                        help="World size (default: 16 for complex navigation)")
    parser.add_argument("--num-objects", type=int, default=20,
                        help="Number of objects/obstacles (default: 20)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--delay", type=float, default=0.15,
                        help="Delay between steps in seconds")
    parser.add_argument("--distance-threshold", type=float, default=2.0,
                        help="Success distance threshold")
    parser.add_argument("--waypoint-threshold", type=float, default=1.0,
                        help="Waypoint arrival threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-model", action="store_true",
                        help="Use greedy policy instead of model")

    args = parser.parse_args()

    # Load model
    model = None
    if not args.no_model:
        model_path = args.model
        if model_path is None:
            candidates = [
                "models/simple/best/best_model.zip",
                "models/simple/final_model.zip",
            ]
            for path in candidates:
                if Path(path).exists():
                    model_path = path
                    break

        if model_path and Path(model_path).exists():
            print(f"Loading model from: {model_path}")
            model = PPO.load(model_path)
        else:
            print("No model found, using greedy policy")

    # Create environment with more objects for complex navigation
    env = SimpleNPCEnv(
        world_size=args.world_size,
        max_steps_per_episode=500,
        distance_threshold=args.distance_threshold,
        seed=args.seed,
        num_obstacles_in_obs=5,
    )

    # Manually set more objects in the world config
    env._world_config.num_objects = args.num_objects

    # Create navigator
    navigator = HybridNavigator(HybridNavigatorConfig(
        waypoint_threshold=args.waypoint_threshold,
        goal_threshold=args.distance_threshold,
        grid_resolution=0.5,
        obstacle_padding=0.15,
        smooth_path=True,
        replan_on_stuck=True,
    ))

    # Create observation builder (matching environment config)
    obs_builder = SimpleObservationBuilder(SimpleObservationConfig(
        world_size=args.world_size,
        num_obstacles=5,
    ))

    print("\n" + "=" * 70)
    print("HYBRID NAVIGATION TEST")
    print("=" * 70)
    print(f"World size: {args.world_size}x{args.world_size}")
    print(f"Num objects: {args.num_objects}")
    print(f"Policy: {'Model' if model else 'Greedy'}")
    print(f"Waypoint threshold: {args.waypoint_threshold}")
    print(f"Goal threshold: {args.distance_threshold}")
    print("=" * 70)
    print("\nPress Ctrl+C to stop\n")

    results = []
    try:
        for ep in range(args.episodes):
            print(f"\n{'#' * 70}")
            print(f"# EPISODE {ep + 1}/{args.episodes}")
            print(f"{'#' * 70}")

            # Reset navigator for new episode
            navigator.reset()

            result = run_episode_hybrid(
                env, model, navigator, obs_builder,
                delay=args.delay,
            )
            results.append(result)

            input("\nPress Enter for next episode...")

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        env.close()

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        successes = sum(1 for r in results if r["success"])
        nav_completes = sum(1 for r in results if r.get("nav_complete", False))
        nav_stucks = sum(1 for r in results if r.get("nav_stuck", False))
        print(f"Episodes: {len(results)}")
        print(f"Successes: {successes}/{len(results)} ({100*successes/len(results):.1f}%)")
        print(f"Nav Complete: {nav_completes}/{len(results)}")
        print(f"Nav Stuck: {nav_stucks}/{len(results)}")
        print(f"Avg Steps: {np.mean([r['steps'] for r in results]):.1f}")
        print(f"Avg Reward: {np.mean([r['reward'] for r in results]):.2f}")


if __name__ == "__main__":
    main()
