"""Quick test to verify combat training setup works."""

from src.combat import CombatEnv, CombatRewardConfig
from src.world import WorldConfig
from src.world.entities import CombatStyle


def test_combat_env_creation():
    """Test that CombatEnv can be created and used."""
    world_config = WorldConfig(size=16, num_objects=0)
    reward_config = CombatRewardConfig(phase=1)

    env = CombatEnv(
        world_config=world_config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        reward_config=reward_config,
        seed=42,
    )

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Info: {info}")

    # Take a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    print("CombatEnv test passed!")


def test_controller_combat():
    """Test that controller handles combat action."""
    from src.controller import NPCController, CombatCommand
    from src.world import World, WorldConfig

    # Create world with enemy
    config = WorldConfig(size=16, num_objects=0)
    world = World(config, seed=42)
    world.reset(num_enemies=1)

    # Create controller
    controller = NPCController()

    # Create mock parsed intent for combat
    class MockIntent:
        type = "task"
        action = "combat"
        target_type = "enemy"
        target = "enemy"
        raw_input = "attack the enemy"

    intent = MockIntent()
    result = controller.process_intent(intent, world.get_state())

    print(f"Controller result type: {type(result).__name__}")
    if isinstance(result, CombatCommand):
        print(f"  Target: {result.target_id}")
        print(f"  Enemy count: {result.enemy_count}")
        print("Controller combat test passed!")
    else:
        print(f"  Unexpected result: {result}")


if __name__ == "__main__":
    test_combat_env_creation()
    print()
    test_controller_combat()
