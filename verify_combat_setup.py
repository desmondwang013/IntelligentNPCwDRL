"""Run this before training to verify combat setup is correct."""

print("Verifying combat training setup...\n")

# Test 1: Import check
print("1. Checking imports...")
try:
    from src.combat import CombatEnv, CombatRewardConfig
    from src.world import WorldConfig, Action
    from src.world.entities import CombatStyle
    print("   OK - All imports successful")
except Exception as e:
    print(f"   FAIL - Import error: {e}")
    exit(1)

# Test 2: Environment creation with close_spawn
print("\n2. Checking CombatEnv with close_spawn...")
try:
    config = WorldConfig(size=16, num_objects=0)
    env = CombatEnv(
        world_config=config,
        num_enemies=1,
        combat_style=CombatStyle.BALANCED,
        max_steps=100,
        seed=42,
        close_spawn=True,
    )
    print(f"   close_spawn attribute: {env.close_spawn}")
    assert env.close_spawn == True, "close_spawn should be True"
    print("   OK - close_spawn=True")
except Exception as e:
    print(f"   FAIL - {e}")
    exit(1)

# Test 3: Close spawn actually works
print("\n3. Checking close spawn distance after reset...")
try:
    obs, info = env.reset()
    npc = env.world.npc
    enemy = env.world.enemies[0]
    dist = npc.position.distance_to(enemy.position)
    print(f"   NPC-Enemy distance: {dist:.3f}")
    print(f"   Attack range: {npc.attack_range}")
    assert dist <= npc.attack_range, f"Enemy should be in attack range! dist={dist}"
    print("   OK - Enemy in attack range")
except Exception as e:
    print(f"   FAIL - {e}")
    exit(1)

# Test 4: Attack attempt bonus in phase 1
print("\n4. Checking attack attempt bonus...")
try:
    phase = env.reward_calculator.config.phase
    bonus = env.reward_calculator.config.phase1_attack_attempt_bonus
    print(f"   Reward phase: {phase}")
    print(f"   Attack attempt bonus: {bonus}")
    assert phase == 1, "Should be phase 1"
    assert bonus > 0, "Should have attack attempt bonus"
    print("   OK - Phase 1 with attack bonus")
except Exception as e:
    print(f"   FAIL - {e}")
    exit(1)

# Test 5: Attack gives positive reward
print("\n5. Checking attack reward...")
try:
    obs, info = env.reset()
    obs, reward, _, _, info = env.step(Action.ATTACK.value)
    breakdown = info['reward_breakdown']
    print(f"   Attack reward: {reward:.3f}")
    print(f"   Breakdown: {breakdown}")
    assert reward > 0, f"Attack should give positive reward, got {reward}"
    assert 'attack_attempt' in breakdown, "Should have attack_attempt in breakdown"
    print("   OK - Attack gives positive reward")
except Exception as e:
    print(f"   FAIL - {e}")
    exit(1)

# Test 6: Run 50 random steps and check total reward
print("\n6. Running 50 random steps...")
try:
    obs, info = env.reset()
    total_reward = 0
    attack_count = 0
    for i in range(50):
        action = env.action_space.sample()
        if action == Action.ATTACK.value:
            attack_count += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Attack count: {attack_count}")
    print(f"   Enemies alive: {info.get('enemies_alive', '?')}")
    if total_reward > 0:
        print("   OK - Positive total reward")
    else:
        print(f"   WARNING - Negative reward ({total_reward:.2f}), but may be OK")
except Exception as e:
    print(f"   FAIL - {e}")
    exit(1)

env.close()

print("\n" + "="*50)
print("ALL CHECKS PASSED - Ready for training!")
print("="*50)
print("\nRun training with:")
print("  python train_combat.py --timesteps 500000")
