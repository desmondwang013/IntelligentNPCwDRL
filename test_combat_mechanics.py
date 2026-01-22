"""Quick test to verify combat mechanics in world.py work correctly."""

from src.world import World, WorldConfig, Action


def test_combat_mechanics():
    """Test basic combat: spawn enemy, attack it, verify damage and death."""
    config = WorldConfig(size=16, num_objects=0)
    world = World(config, seed=42)
    world.reset(num_enemies=1)

    print("=== Combat Mechanics Test ===\n")

    # Verify enemy spawned
    assert len(world.enemies) == 1, "Should have 1 enemy"
    enemy = world.enemies[0]
    print(f"Enemy spawned: {enemy.entity_id}")
    print(f"  Position: ({enemy.position.x:.2f}, {enemy.position.y:.2f})")
    print(f"  Health: {enemy.health}/{enemy.max_health}")
    print(f"  Is aggro: {enemy.is_aggro}")

    # Move NPC close to enemy for attack
    world.npc.position = enemy.position.copy()
    world.npc.position.x -= 0.3  # Just inside attack range (0.5)
    print(f"\nNPC moved to ({world.npc.position.x:.2f}, {world.npc.position.y:.2f})")
    print(f"  Distance to enemy: {world.npc.position.distance_to(enemy.position):.2f}")
    print(f"  NPC attack range: {world.npc.attack_range}")

    # Attack the enemy
    print("\n--- NPC attacks ---")
    state = world.step(Action.ATTACK)
    print(f"  Damage dealt: {state['last_damage_dealt']}")
    print(f"  Enemy health: {enemy.health}/{enemy.max_health}")
    print(f"  Enemy is aggro: {enemy.is_aggro}")
    assert state['last_damage_dealt'] == 1, "Should deal 1 damage"
    assert enemy.is_aggro, "Enemy should be aggro after taking damage"

    # Enemy should attack back (after cooldown)
    print("\n--- Waiting for cooldowns ---")
    for i in range(8):  # Wait for cooldowns
        state = world.step(Action.WAIT)
        if state['last_damage_taken'] > 0:
            print(f"  Tick {i+1}: NPC took {state['last_damage_taken']} damage")

    print(f"\nNPC health after enemy attacks: {world.npc.health}/{world.npc.max_health}")

    # Kill the enemy
    print("\n--- Finishing off enemy ---")
    while enemy.is_alive:
        # Wait for NPC cooldown
        while not world.npc.can_attack:
            world.step(Action.WAIT)
        state = world.step(Action.ATTACK)
        if state['last_damage_dealt'] > 0:
            print(f"  Hit! Enemy health: {enemy.health}/{enemy.max_health}")
        if state['last_enemy_killed']:
            print(f"  Enemy killed: {state['last_enemy_killed']}")

    assert not enemy.is_alive, "Enemy should be dead"
    assert state['last_enemy_killed'] == enemy.entity_id, "Should report killed enemy"

    print("\n=== All combat tests passed! ===")


def test_multiple_enemies():
    """Test combat with multiple enemies."""
    config = WorldConfig(size=32, num_objects=0)
    world = World(config, seed=42)
    world.reset(num_enemies=3)

    print("\n=== Multiple Enemies Test ===\n")
    print(f"Spawned {len(world.enemies)} enemies")
    for e in world.enemies:
        dist = world.npc.position.distance_to(e.position)
        print(f"  {e.entity_id}: distance={dist:.2f}, health={e.health}")

    # Find nearest and attack
    nearest = world.get_nearest_enemy(world.npc.position)
    print(f"\nNearest enemy: {nearest.entity_id}")

    # Move NPC next to nearest enemy
    world.npc.position = nearest.position.copy()
    world.npc.position.x -= 0.3

    # Attack
    state = world.step(Action.ATTACK)
    print(f"Attacked nearest enemy, dealt {state['last_damage_dealt']} damage")
    print(f"Enemy {nearest.entity_id} now has {nearest.health} health")

    alive = world.get_alive_enemies()
    print(f"Alive enemies: {len(alive)}")

    print("\n=== Multiple enemies test passed! ===")


if __name__ == "__main__":
    test_combat_mechanics()
    test_multiple_enemies()
