"""
Demo script to validate the world module works correctly.
Run from project root: python demo_world.py
"""
import json
from src.world import World
from src.world.world import Action


def print_state(state: dict, verbose: bool = False) -> None:
    print(f"\n=== Tick {state['tick']} ===")
    print(f"NPC: ({state['npc']['position']['x']:.2f}, {state['npc']['position']['y']:.2f})")
    print(f"User: ({state['user']['position']['x']:.2f}, {state['user']['position']['y']:.2f})")
    print(f"Last action: {state['last_action']}")
    if state['last_speech']:
        print(f"NPC said: {state['last_speech']}")

    if verbose:
        print("\nObjects:")
        for obj in state['objects']:
            print(f"  {obj['entity_id']}: {obj['color']} {obj['size']} {obj['shape']} "
                  f"at ({obj['position']['x']:.2f}, {obj['position']['y']:.2f}) "
                  f"[{obj['object_type']}]")


def main():
    print("=== World Module Demo ===\n")

    # Create world with fixed seed for reproducibility
    world = World(seed=42)
    state = world.get_state()

    print("Initial state:")
    print_state(state, verbose=True)

    # Demonstrate movement
    print("\n--- Moving NPC ---")
    actions = [
        (Action.MOVE_RIGHT, None),
        (Action.MOVE_RIGHT, None),
        (Action.MOVE_UP, None),
        (Action.WAIT, None),
        (Action.SPEAK, "Hello, I am the NPC!"),
        (Action.MOVE_LEFT, None),
    ]

    for action, speech in actions:
        state = world.step(action.value, speech)
        print_state(state)

    # Demonstrate nearest objects
    print("\n--- Nearest Objects to NPC ---")
    nearest = world.get_nearest_objects(world.npc.position, n=3)
    for obj in nearest:
        dist = world.npc.position.distance_to(obj.position)
        print(f"  {obj.entity_id}: {obj.color.name} {obj.shape.name} at distance {dist:.2f}")

    # Demonstrate state serialization (for Unity)
    print("\n--- Serialized State (JSON) ---")
    state = world.get_state()
    print(json.dumps(state, indent=2)[:500] + "...")

    # Demonstrate reset
    print("\n--- Reset with new seed ---")
    world.reset(seed=123)
    state = world.get_state()
    print_state(state)

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
