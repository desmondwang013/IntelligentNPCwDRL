"""
Demo script to validate the reward system.
Run from project root: python demo_reward.py
"""
from src.runtime import Runtime, RuntimeConfig
from src.reward import RewardConfig
from src.world.world import Action
from src.controller import IntentType


def main():
    print("=== Reward System Demo ===\n")

    # Create runtime with default reward config
    config = RuntimeConfig(world_seed=42)
    runtime = Runtime(config=config)

    # Show reward configuration
    print("Reward Configuration:")
    for key, value in runtime.reward_calculator.get_config().items():
        print(f"  {key}: {value}")
    print()

    # Get initial state
    state = runtime.get_state()
    target_obj = state["world"]["objects"][0]
    print(f"Target: {target_obj['entity_id']} - {target_obj['color']} "
          f"{target_obj['shape']} at ({target_obj['position']['x']:.1f}, "
          f"{target_obj['position']['y']:.1f})")

    npc = state["world"]["npc"]["position"]
    print(f"NPC starts at: ({npc['x']:.1f}, {npc['y']:.1f})")
    print()

    # Submit instruction
    runtime.submit_instruction(
        text=f"Go to the {target_obj['color']} {target_obj['shape']}",
        intent_type=IntentType.MOVE_TO_OBJECT,
        target_entity_id=target_obj["entity_id"],
        distance_threshold=2.0,
    )

    # Run simulation and track rewards
    print("=== Simulation with Rewards ===")
    print(f"{'Tick':<6} {'Action':<12} {'Reward':>8} {'Progress':>9} {'Time':>7} {'Collision':>10} {'Total':>8}")
    print("-" * 70)

    obs = runtime.get_observation()
    total_reward = 0.0

    # Simple policy: move toward target (right and up based on seed 42 layout)
    # Target obj_0 is at ~(46, 42), NPC at ~(18, 15)
    # So we need to go RIGHT and UP mostly

    for i in range(30):
        # Simple heuristic policy
        npc_pos = runtime.world.npc.position
        target_x, target_y = target_obj['position']['x'], target_obj['position']['y']

        dx = target_x - npc_pos.x
        dy = target_y - npc_pos.y

        if abs(dx) > abs(dy):
            action = Action.MOVE_RIGHT.value if dx > 0 else Action.MOVE_LEFT.value
        else:
            action = Action.MOVE_UP.value if dy > 0 else Action.MOVE_DOWN.value

        result = runtime.step(action)
        total_reward += result.reward

        info = result.reward_info
        print(f"{result.tick:<6} {Action(result.npc_action).name:<12} "
              f"{result.reward:>8.3f} {info.progress:>9.3f} {info.time:>7.3f} "
              f"{info.collision:>10.3f} {total_reward:>8.3f}")

        # Check for completion
        for event in result.events:
            if event.event_type.name == "INTENT_COMPLETED":
                print(f"\n*** INTENT COMPLETED at tick {result.tick}! ***")
                print(f"Completion bonus: {info.completion:.1f}")

    print()

    # Test oscillation penalty
    print("=== Testing Oscillation Penalty ===")
    runtime.reset(seed=42)
    runtime.submit_instruction(
        text="Go somewhere",
        intent_type=IntentType.GENERIC,
        target_entity_id=target_obj["entity_id"],
    )

    # Deliberately oscillate
    actions = [Action.MOVE_RIGHT, Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_LEFT]
    for action in actions:
        result = runtime.step(action.value)
        print(f"Action: {action.name:<12} Oscillation penalty: {result.reward_info.oscillation:.3f}")
    print()

    # Test collision penalty
    print("=== Testing Collision Penalty ===")
    runtime.reset(seed=42)

    # Move to edge of world
    for _ in range(50):
        runtime.step(Action.MOVE_LEFT.value)

    # Now try to move further left (should collide with boundary)
    result = runtime.step(Action.MOVE_LEFT.value)
    print(f"Tried to move into boundary - Collision penalty: {result.reward_info.collision:.3f}")
    print()

    # Test timeout penalty
    print("=== Testing Timeout (shortened for demo) ===")
    short_timeout_config = RuntimeConfig(
        world_seed=42,
        default_intent_timeout=20,  # Very short timeout for demo
    )
    runtime2 = Runtime(config=short_timeout_config)

    runtime2.submit_instruction(
        text="Go far away",
        intent_type=IntentType.MOVE_TO_OBJECT,
        target_entity_id="obj_5",  # Some distant object
        distance_threshold=1.0,
    )

    # Just wait and let it timeout
    for i in range(25):
        result = runtime2.step(Action.WAIT.value)
        if result.reward_info.completion != 0:
            print(f"Tick {result.tick}: Timeout penalty = {result.reward_info.completion:.1f}")
            break

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
