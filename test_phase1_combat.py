"""Test phase 1 combat changes: close spawn + attack attempt bonus."""

from src.combat import CombatEnv, CombatRewardConfig
from src.world import WorldConfig, Action
from src.world.entities import CombatStyle


def test_close_spawn():
    """Test that close spawn puts enemies within attack range."""
    print("=== Close Spawn Test ===\n")

    config = WorldConfig(size=16, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        seed=42,
        close_spawn=True,
    )

    # Test multiple resets
    for i in range(5):
        obs, info = env.reset(seed=42 + i)
        npc = env.world.npc
        enemy = env.world.enemies[0]
        dist = npc.position.distance_to(enemy.position)
        in_range = dist <= npc.attack_range

        print(f"Reset {i+1}: NPC at ({npc.position.x:.2f}, {npc.position.y:.2f})")
        print(f"         Enemy at ({enemy.position.x:.2f}, {enemy.position.y:.2f})")
        print(f"         Distance: {dist:.3f}, Attack range: {npc.attack_range}")
        print(f"         In range: {in_range}")
        assert in_range, f"Enemy should be within attack range! dist={dist}"

    env.close()
    print("\nClose spawn test passed!")


def test_attack_attempt_bonus():
    """Test that attack attempts give bonus in phase 1."""
    print("\n=== Attack Attempt Bonus Test ===\n")

    config = WorldConfig(size=16, num_objects=0)
    reward_config = CombatRewardConfig(phase=1)

    env = CombatEnv(
        world_config=config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        reward_config=reward_config,
        seed=42,
        close_spawn=True,
    )

    obs, info = env.reset()

    # Take WAIT action
    obs, reward_wait, _, _, info_wait = env.step(Action.WAIT.value)
    print(f"WAIT reward: {reward_wait:.3f}")
    print(f"  Breakdown: {info_wait['reward_breakdown']}")

    # Take ATTACK action (should include attempt bonus)
    obs, reward_attack, _, _, info_attack = env.step(Action.ATTACK.value)
    print(f"\nATTACK reward: {reward_attack:.3f}")
    print(f"  Breakdown: {info_attack['reward_breakdown']}")

    # Attack should have higher reward due to:
    # 1. Attack attempt bonus (+0.1)
    # 2. Damage dealt (+1.0 if hit)
    assert reward_attack > reward_wait, "Attack should give higher reward than wait"
    assert 'attack_attempt' in info_attack['reward_breakdown'], "Should have attack_attempt bonus"

    env.close()
    print("\nAttack attempt bonus test passed!")


def test_phase_transition():
    """Test that phase 2 disables bonuses."""
    print("\n=== Phase Transition Test ===\n")

    config = WorldConfig(size=16, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        seed=42,
        close_spawn=True,
    )

    obs, _ = env.reset()

    # Phase 1: attack attempt bonus
    env.set_reward_phase(1)
    obs, _, _, _, info1 = env.step(Action.ATTACK.value)
    has_bonus_p1 = 'attack_attempt' in info1['reward_breakdown']
    print(f"Phase 1 - Attack attempt bonus: {has_bonus_p1}")

    # Reset and transition to phase 2
    env.set_reward_phase(2)
    env.set_close_spawn(False)
    obs, _ = env.reset()

    obs, _, _, _, info2 = env.step(Action.ATTACK.value)
    has_bonus_p2 = 'attack_attempt' in info2['reward_breakdown']
    print(f"Phase 2 - Attack attempt bonus: {has_bonus_p2}")

    assert has_bonus_p1, "Phase 1 should have attack attempt bonus"
    assert not has_bonus_p2, "Phase 2 should NOT have attack attempt bonus"

    env.close()
    print("\nPhase transition test passed!")


if __name__ == "__main__":
    test_close_spawn()
    test_attack_attempt_bonus()
    test_phase_transition()
    print("\n=== All tests passed! ===")
