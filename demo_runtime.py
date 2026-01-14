"""
Demo script to validate the runtime loop.
Run from project root: python demo_runtime.py

NOTE: Requires sentence-transformers for full functionality.
      pip install sentence-transformers
"""
import random
from src.runtime import Runtime, RuntimeConfig
from src.world.world import Action
from src.intent import IntentType


def random_policy(observation):
    """A random policy for testing. Replace with trained policy later."""
    return random.choice([
        Action.MOVE_UP.value,
        Action.MOVE_DOWN.value,
        Action.MOVE_LEFT.value,
        Action.MOVE_RIGHT.value,
        Action.WAIT.value,
    ])


def main():
    print("=== Runtime Loop Demo ===\n")

    try:
        # Create runtime
        config = RuntimeConfig(world_seed=42)
        runtime = Runtime(config=config)

        print(f"Observation dimension: {runtime.observation_dim}")
        print(f"Initial tick: {runtime.tick}")
        print()

        # Get initial state
        state = runtime.get_state()
        print("=== Initial State ===")
        npc = state["world"]["npc"]["position"]
        user = state["world"]["user"]["position"]
        print(f"NPC: ({npc['x']:.2f}, {npc['y']:.2f})")
        print(f"User: ({user['x']:.2f}, {user['y']:.2f})")
        print(f"Active intent: {state['intent']['has_intent']}")
        print()

        # Submit an instruction
        print("=== Submitting Instruction ===")
        target_obj = state["world"]["objects"][0]
        print(f"Target: {target_obj['entity_id']} - {target_obj['color']} "
              f"{target_obj['shape']} at ({target_obj['position']['x']:.2f}, "
              f"{target_obj['position']['y']:.2f})")

        runtime.submit_instruction(
            text=f"Go to the {target_obj['color']} {target_obj['shape']}",
            intent_type=IntentType.MOVE_TO_OBJECT,
            target_entity_id=target_obj["entity_id"],
            distance_threshold=2.0,
        )

        # Run simulation loop
        print("\n=== Running Simulation (50 ticks) ===")
        obs = runtime.get_observation()

        for i in range(50):
            # Policy selects action
            action = random_policy(obs)

            # Step the simulation
            result = runtime.step(action)
            obs = result.observation

            # Print status every 10 ticks
            if (i + 1) % 10 == 0 or result.events:
                npc_pos = result.world_state["npc"]["position"]
                intent_info = result.intent_state

                print(f"Tick {result.tick}: "
                      f"NPC=({npc_pos['x']:.1f}, {npc_pos['y']:.1f}), "
                      f"action={Action(result.npc_action).name}, "
                      f"intent_active={intent_info['has_intent']}")

                # Print any events
                for event in result.events:
                    print(f"  EVENT: {event.event_type.name} - {event.data}")

        # Test interruption
        print("\n=== Testing Interruption ===")
        runtime.submit_instruction(
            text="Actually, come to me instead",
            intent_type=IntentType.MOVE_TO_USER,
            target_entity_id="user",
        )

        # Run a few more ticks
        for i in range(10):
            action = random_policy(obs)
            result = runtime.step(action)
            obs = result.observation

            if result.events:
                for event in result.events:
                    print(f"Tick {result.tick} EVENT: {event.event_type.name}")

        # Test user movement
        print("\n=== Testing User Movement ===")
        state_before = runtime.get_state()
        user_before = state_before["world"]["user"]["position"]
        print(f"User before: ({user_before['x']:.2f}, {user_before['y']:.2f})")

        runtime.move_user(5.0, 5.0)  # queue movement
        result = runtime.step(Action.WAIT.value)  # process it

        user_after = result.world_state["user"]["position"]
        print(f"User after:  ({user_after['x']:.2f}, {user_after['y']:.2f})")

        # Test cancellation
        print("\n=== Testing Cancellation ===")
        runtime.cancel_instruction(reason="user_changed_mind")
        result = runtime.step(Action.WAIT.value)
        for event in result.events:
            print(f"EVENT: {event.event_type.name} - {event.data}")

        # Final state
        print("\n=== Final State ===")
        final_state = runtime.get_state()
        print(f"Total ticks: {final_state['tick']}")
        print(f"Intent history: {len(runtime.intent_manager.get_history())} intents")
        print(f"Active intent: {final_state['intent']['has_intent']}")

        # Show intent history
        print("\n=== Intent History ===")
        for intent in runtime.intent_manager.get_history():
            print(f"  {intent['intent_id']}: {intent['status']} - \"{intent['text'][:40]}...\"")

        print("\n=== Demo Complete ===")

    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nTo run this demo, install sentence-transformers:")
        print("  pip install sentence-transformers")


if __name__ == "__main__":
    main()
