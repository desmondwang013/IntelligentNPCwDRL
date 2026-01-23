from src.world import Action
print("Actions:", [(a.name, a.value) for a in Action])
print("\nCombatEnv uses Discrete(6), so valid actions are 0-5:")
for a in Action:
    if a.value < 6:
        print(f"  {a.value}: {a.name}")
    else:
        print(f"  {a.value}: {a.name} (NOT in action space)")
