"""
LEGACY DEMO: Observation builder with embeddings (575+ dimensions).

In the current architecture, use SimpleObservationBuilder (24 dimensions)
which contains only structured goal information, no embeddings.

For the current architecture, see SimpleObservationBuilder in:
  src/observation/simple_builder.py

Original description:
Demo script to validate the observation builder.
Run from project root: python demo_observation.py

This demo uses a dummy embedding, so it works without sentence-transformers.
"""
import numpy as np
from src.world import World
from src.observation import ObservationBuilder


def main():
    print("=== Observation Builder Demo ===\n")

    # Create world
    world = World(seed=42)
    state = world.get_state()

    # Create observation builder
    obs_builder = ObservationBuilder()

    print(f"Observation dimension: {obs_builder.observation_dim}")
    print()

    # Create a dummy intent embedding (normally from IntentManager)
    dummy_embedding = np.random.randn(384).astype(np.float32)
    dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)  # normalize

    # Build observation
    obs = obs_builder.build(
        world_state=state,
        intent_embedding=dummy_embedding,
        intent_age_ticks=32,  # 2 seconds
        focus_hint="obj_0",
    )

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print()

    # Show observation structure
    spec = obs_builder.get_observation_spec()
    print("=== Observation Structure ===")
    for name, info in spec["structure"].items():
        start = info["start"]
        size = info["size"]
        end = start + size
        sample = obs[start:end]

        if size <= 8:
            print(f"{name}: [{start}:{end}] = {sample}")
        else:
            print(f"{name}: [{start}:{end}] shape={sample.shape}, "
                  f"range=[{sample.min():.3f}, {sample.max():.3f}]")
    print()

    # Decode some values for verification
    print("=== Decoded Values ===")
    npc_x = obs[0] * 64
    npc_y = obs[1] * 64
    user_x = obs[2] * 64
    user_y = obs[3] * 64
    print(f"NPC position: ({npc_x:.2f}, {npc_y:.2f})")
    print(f"User position: ({user_x:.2f}, {user_y:.2f})")

    # Check focus hint
    focus_start = spec["structure"]["focus_hint"]["start"]
    focus_vec = obs[focus_start:focus_start + 8]
    focus_idx = np.argmax(focus_vec) if focus_vec.max() > 0 else -1
    print(f"Focus hint index: {focus_idx} (obj_0 should be in nearest 8)")
    print()

    # Test multiple observations over time
    print("=== Observations Over 5 Ticks ===")
    from src.world.world import Action

    for i in range(5):
        state = world.step(Action.MOVE_RIGHT.value)
        obs = obs_builder.build(
            world_state=state,
            intent_embedding=dummy_embedding,
            intent_age_ticks=32 + i + 1,
            focus_hint=None,
        )
        npc_x = obs[0] * 64
        intent_age = obs[spec["structure"]["intent_age"]["start"]]
        print(f"Tick {state['tick']}: NPC x={npc_x:.2f}, intent_age_norm={intent_age:.4f}")

    print()

    # Print full object feature breakdown
    print("=== Per-Object Feature Breakdown ===")
    for name, size in spec["object_feature_breakdown"].items():
        print(f"  {name}: {size}")
    print(f"  Total per object: {sum(spec['object_feature_breakdown'].values())}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
