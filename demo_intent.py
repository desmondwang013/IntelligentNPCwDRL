"""
Demo script to validate the intent system.
Run from project root: python demo_intent.py

NOTE: Requires sentence-transformers to be installed for embeddings.
      pip install sentence-transformers
"""
from src.world import World
from src.world.world import Action
from src.intent import IntentManager, IntentType
from src.intent.intent import CompletionCriteria


def main():
    print("=== Intent System Demo ===\n")

    # Create world
    world = World(seed=42)
    state = world.get_state()

    print(f"NPC starting at: ({state['npc']['position']['x']:.2f}, "
          f"{state['npc']['position']['y']:.2f})")

    # Pick a target object
    target_obj = state['objects'][0]
    print(f"Target object: {target_obj['entity_id']} ({target_obj['color']} "
          f"{target_obj['shape']}) at ({target_obj['position']['x']:.2f}, "
          f"{target_obj['position']['y']:.2f})")

    # Create intent manager
    print("\nInitializing IntentManager (loading embedding model)...")
    try:
        manager = IntentManager()

        # Test 1: Create an intent
        print("\n--- Test 1: Create Intent ---")
        intent = manager.new_intent(
            text=f"Go to the {target_obj['color']} {target_obj['shape']}",
            current_tick=state['tick'],
            intent_type=IntentType.MOVE_TO_OBJECT,
            criteria=CompletionCriteria(
                target_entity_id=target_obj['entity_id'],
                distance_threshold=2.0,
            ),
            focus_hint=target_obj['entity_id'],
        )
        print(f"Created intent: {intent.intent_id}")
        print(f"Text: '{intent.text}'")
        print(f"Embedding shape: {intent.embedding.shape}")
        print(f"Embedding sample: {intent.embedding[:5]}...")

        # Test 2: Simulate some ticks
        print("\n--- Test 2: Simulate Ticks ---")
        for i in range(5):
            # Move NPC right (toward nothing in particular)
            state = world.step(Action.MOVE_RIGHT.value)
            event = manager.update(state['tick'], state)

            intent_state = manager.get_state(state['tick'])
            print(f"Tick {state['tick']}: age={intent_state['age_ticks']}, "
                  f"event={event.event_type if event else 'none'}")

        # Test 3: Preemption
        print("\n--- Test 3: Intent Preemption ---")
        new_intent = manager.new_intent(
            text="Actually, come to me instead",
            current_tick=state['tick'],
            intent_type=IntentType.MOVE_TO_USER,
            criteria=CompletionCriteria(
                target_entity_id="user",
                distance_threshold=2.0,
            ),
        )
        print(f"New intent created: {new_intent.intent_id}")
        print(f"Previous intent was canceled (preempted)")

        # Check history
        history = manager.get_history()
        print(f"Intent history: {len(history)} completed/canceled intents")
        for h in history:
            print(f"  - {h['intent_id']}: {h['status']}")

        # Test 4: Embedding cache
        print("\n--- Test 4: Embedding Cache ---")
        print(f"Cache size: {manager.embedder.cache_size}")

        # Embed the same text again (should hit cache)
        emb1 = manager.embedder.embed("Go to the red triangle")
        emb2 = manager.embedder.embed("Go to the red triangle")
        print(f"Same text embeddings equal: {(emb1 == emb2).all()}")
        print(f"Cache size after: {manager.embedder.cache_size}")

        # Test 5: Event log
        print("\n--- Test 5: Event Log ---")
        events = manager.get_event_log()
        for e in events:
            print(f"  {e['event_type']}: {e['intent_id']} at tick {e['tick']}")

        print("\n=== Demo Complete ===")

    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nTo run this demo, install sentence-transformers:")
        print("  pip install sentence-transformers")
        print("\nThe intent system code is ready, but embeddings require this library.")


if __name__ == "__main__":
    main()
