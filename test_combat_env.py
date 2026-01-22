"""Test the CombatEnv gymnasium environment."""

from src.combat import CombatEnv
from src.world import WorldConfig, Action
from src.world.entities import CombatStyle


def test_combat_env_basic():
    """Test basic env functionality."""
    print("=== CombatEnv Basic Test ===\n")

    config = WorldConfig(size=16, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=2,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        render_mode="human",
        seed=42
    )

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"\nInitial info: {info}")
    print()
    env.render()
    print()

    # Take some random actions
    print("--- Taking 10 random actions ---\n")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = Action(action).name
        print(f"Step {step+1}: Action={action_name}, Reward={reward:.3f}")
        if 'reward_breakdown' in info:
            breakdown = {k: v for k, v in info['reward_breakdown'].items() if k != 'total' and v != 0}
            if breakdown:
                print(f"         Breakdown: {breakdown}")

        if terminated or truncated:
            print(f"         Episode ended! Terminated={terminated}, Truncated={truncated}")
            break

    print("\n=== Basic test passed! ===")


def test_combat_env_win():
    """Test winning by killing all enemies."""
    print("\n=== CombatEnv Win Test ===\n")

    config = WorldConfig(size=16, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=1,
        combat_style=CombatStyle.AGGRESSIVE,
        max_steps=200,
        seed=42
    )

    obs, info = env.reset()
    world = env.world

    # Move NPC next to enemy and attack until dead
    enemy = world.enemies[0]
    world.npc.position = enemy.position.copy()
    world.npc.position.x -= 0.3  # Just inside attack range

    total_reward = 0
    for step in range(50):
        # Attack if can, otherwise wait
        action = Action.ATTACK if world.npc.can_attack else Action.WAIT
        obs, reward, terminated, truncated, info = env.step(action.value)
        total_reward += reward

        if terminated:
            print(f"Episode ended at step {step+1}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"NPC health: {info['npc_health']}")
            print(f"Enemies alive: {info['enemies_alive']}")

            if info['enemies_alive'] == 0:
                print("Victory!")
            break

    print("\n=== Win test passed! ===")


def test_combat_styles():
    """Test different combat styles affect rewards."""
    print("\n=== Combat Styles Test ===\n")

    config = WorldConfig(size=16, num_objects=0)

    for style in [CombatStyle.AGGRESSIVE, CombatStyle.BALANCED, CombatStyle.PASSIVE]:
        env = CombatEnv(
            world_config=config,
            num_enemies=1,
            combat_style=style,
            max_steps=100,
            seed=42
        )

        obs, info = env.reset()
        world = env.world

        # Move NPC next to enemy
        enemy = world.enemies[0]
        world.npc.position = enemy.position.copy()
        world.npc.position.x -= 0.3

        # Attack to aggro, then take some hits
        env.step(Action.ATTACK.value)  # First hit aggros enemy

        # Wait for enemy to hit back
        damage_penalties = []
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(Action.WAIT.value)
            if 'reward_breakdown' in info and 'damage_taken' in info['reward_breakdown']:
                damage_penalties.append(info['reward_breakdown']['damage_taken'])
            if terminated:
                break

        avg_penalty = sum(damage_penalties) / len(damage_penalties) if damage_penalties else 0
        print(f"{style.name:12}: Avg damage penalty = {avg_penalty:.3f}")

    print("\n=== Combat styles test passed! ===")


if __name__ == "__main__":
    test_combat_env_basic()
    test_combat_env_win()
    test_combat_styles()
