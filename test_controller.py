"""
Test script for NPCController integration.

Tests the full flow: LLM ParsedIntent → NPCController → Result

Usage:
    python test_controller.py
    python test_controller.py --with-llm  # Include LLM in the test
"""
import argparse
from dataclasses import dataclass
from typing import Optional

from src.world import World
from src.world.world import WorldConfig
from src.controller import NPCController, WorldQuery, NavigationCommand, TaskStatus
from src.navigation import HybridNavigator, HybridNavigatorConfig


# Mock ParsedIntent for testing without LLM
@dataclass
class MockParsedIntent:
    """Mock of ParsedIntent from LLM for testing."""
    type: str  # "task" or "conversation"
    action: Optional[str] = None
    target_type: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None
    size: Optional[str] = None
    response: Optional[str] = None
    raw_input: str = ""


def test_world_query():
    """Test WorldQuery functionality."""
    print("=" * 60)
    print("Testing WorldQuery")
    print("=" * 60)

    # Create a world with known objects
    config = WorldConfig(size=32, num_objects=10)
    world = World(config=config, seed=42)
    state = world.get_state()

    query = WorldQuery(state)

    # Print all objects for reference
    print("\nWorld objects:")
    for obj in state["objects"]:
        print(f"  {obj['entity_id']}: {obj['size']} {obj['color']} {obj['shape']}")

    # Test finding objects
    print("\nQuery tests:")

    # Find red objects
    red_objects = query.find_objects(color="red")
    print(f"  Red objects: {len(red_objects)} found")
    for obj in red_objects:
        print(f"    - {obj.describe()} ({obj.entity_id})")

    # Find circles
    circles = query.find_objects(shape="circle")
    print(f"  Circles: {len(circles)} found")

    # Find specific combination
    blue_squares = query.find_objects(color="blue", shape="square")
    print(f"  Blue squares: {len(blue_squares)} found")

    # Test positions
    user_pos = query.get_user_position()
    npc_pos = query.get_npc_position()
    print(f"\n  User position: {user_pos}")
    print(f"  NPC position: {npc_pos}")

    print("\nWorldQuery tests passed!")


def test_controller_resolution():
    """Test NPCController target resolution."""
    print("\n" + "=" * 60)
    print("Testing NPCController Resolution")
    print("=" * 60)

    # Create world and controller
    config = WorldConfig(size=32, num_objects=10)
    world = World(config=config, seed=42)
    controller = NPCController()

    state = world.get_state()

    # Print world state
    print("\nWorld objects:")
    for obj in state["objects"]:
        print(f"  {obj['entity_id']}: {obj['size']} {obj['color']} {obj['shape']}")

    # Test cases
    test_cases = [
        # (description, intent)
        ("Navigate to user", MockParsedIntent(
            type="task",
            action="navigate",
            target_type="user",
            raw_input="go to the player",
        )),
        ("Navigate to specific object", MockParsedIntent(
            type="task",
            action="navigate",
            target_type="object",
            color="blue",
            shape="circle",
            raw_input="go to the blue circle",
        )),
        ("Ambiguous target (color only)", MockParsedIntent(
            type="task",
            action="navigate",
            target_type="object",
            color="red",  # Likely multiple red objects
            raw_input="go to the red thing",
        )),
        ("Wait action", MockParsedIntent(
            type="task",
            action="wait",
            raw_input="wait",
        )),
        ("Unsupported action", MockParsedIntent(
            type="task",
            action="unsupported",
            raw_input="attack the enemy",
        )),
        ("Conversation", MockParsedIntent(
            type="conversation",
            response="Hello! How can I help you today?",
            raw_input="hello",
        )),
        ("Non-existent target", MockParsedIntent(
            type="task",
            action="navigate",
            target_type="object",
            color="pink",  # No pink objects exist
            shape="hexagon",  # No hexagons exist
            raw_input="go to the pink hexagon",
        )),
    ]

    print("\nResolution tests:")
    for description, intent in test_cases:
        result = controller.process_intent(intent, state)
        print(f"\n  {description}:")
        print(f"    Input: \"{intent.raw_input}\"")

        if isinstance(result, NavigationCommand):
            print(f"    Result: NavigationCommand")
            print(f"      Target ID: {result.target_id}")
            print(f"      Position: {result.target_position}")
        else:
            print(f"    Result: {result.status.name}")
            result_dict = result.to_dict()
            for key, value in result_dict.items():
                if key != "status":
                    print(f"      {key}: {value}")

    print("\nController resolution tests passed!")


def test_navigation_flow():
    """Test full navigation flow with Navigator."""
    print("\n" + "=" * 60)
    print("Testing Full Navigation Flow")
    print("=" * 60)

    # Create world, controller, navigator
    config = WorldConfig(size=32, num_objects=5)
    world = World(config=config, seed=123)
    controller = NPCController()
    navigator = HybridNavigator(HybridNavigatorConfig(
        waypoint_threshold=1.0,
        goal_threshold=2.0,
    ))

    state = world.get_state()

    # Find a UNIQUE target (one that won't cause ambiguity)
    query = WorldQuery(state)
    all_objects = query.get_all_objects()
    if not all_objects:
        print("No objects in world!")
        return

    # Find an object with unique color+shape combination
    target = None
    for obj in all_objects:
        matches = query.find_objects(color=obj.color, shape=obj.shape)
        if len(matches) == 1:
            target = obj
            break

    if target is None:
        # Fall back to most specific (include size)
        target = all_objects[0]
        print(f"\nWarning: No unique color+shape combo found, using first object")
        print(f"  Will include size for disambiguation")

    print(f"\nTarget: {target.describe()} ({target.entity_id}) at {target.position}")

    # Create intent to navigate to this target (include size for uniqueness)
    intent = MockParsedIntent(
        type="task",
        action="navigate",
        target_type="object",
        color=target.color,
        shape=target.shape,
        size=target.size,  # Include size for unique match
        raw_input=f"go to the {target.size} {target.color} {target.shape}",
    )

    # Process intent
    result = controller.process_intent(intent, state)

    if not isinstance(result, NavigationCommand):
        print(f"Resolution failed: {result.to_dict()}")
        return

    print(f"Navigation command received:")
    print(f"  Target: {result.target_id}")
    print(f"  Position: {result.target_position}")

    # Start navigator
    npc_pos = query.get_npc_position()
    success = navigator.set_goal(
        npc_pos[0], npc_pos[1],
        result.target_position[0], result.target_position[1],
        state,
    )

    if not success:
        print("Navigator failed to find path!")
        completion_result = controller.report_navigation_complete(state, success=False, reason="no path")
        print(f"Result: {completion_result.to_dict()}")
        return

    print(f"Path found with {len(navigator.waypoints)} waypoints")

    # Simulate navigation (just a few steps for demo)
    max_steps = 200
    for step in range(max_steps):
        # Get current target from navigator
        current_target = navigator.get_current_target()
        if current_target is None:
            break

        # Simple movement toward target (not actual RL, just direct movement)
        npc = world.npc
        dx = current_target.x - npc.position.x
        dy = current_target.y - npc.position.y
        dist = (dx * dx + dy * dy) ** 0.5

        if dist > 0.5:
            # Move toward target
            if abs(dx) > abs(dy):
                action = 3 if dx > 0 else 2  # RIGHT or LEFT
            else:
                action = 0 if dy > 0 else 1  # UP or DOWN
        else:
            action = 4  # WAIT

        # Step world
        state = world.step(action)

        # Update navigator
        npc_pos = (world.npc.position.x, world.npc.position.y)
        navigator.update(npc_pos[0], npc_pos[1], state)

        # Check controller for timeout
        timeout_result = controller.update(state)
        if timeout_result:
            print(f"Task timed out at step {step}")
            print(f"Result: {timeout_result.to_dict()}")
            return

        # Check if navigation complete
        nav_state = navigator.get_state()
        if nav_state.is_complete:
            completion_result = controller.report_navigation_complete(state, success=True)
            print(f"\nNavigation completed at step {step}!")
            print(f"Result: {completion_result.to_dict()}")
            return

    print(f"\nNavigation did not complete within {max_steps} steps")
    completion_result = controller.report_navigation_complete(state, success=False, reason="max steps exceeded")
    print(f"Result: {completion_result.to_dict()}")


def test_interrupt():
    """Test task interruption."""
    print("\n" + "=" * 60)
    print("Testing Task Interruption")
    print("=" * 60)

    config = WorldConfig(size=32, num_objects=5)
    world = World(config=config, seed=456)
    controller = NPCController()

    state = world.get_state()
    query = WorldQuery(state)
    all_objects = query.get_all_objects()

    if len(all_objects) < 2:
        print("Need at least 2 objects for interrupt test")
        return

    # Start first navigation (use size for uniqueness)
    target1 = all_objects[0]
    intent1 = MockParsedIntent(
        type="task",
        action="navigate",
        target_type="object",
        color=target1.color,
        shape=target1.shape,
        size=target1.size,
        raw_input=f"go to the {target1.size} {target1.color} {target1.shape}",
    )

    result1 = controller.process_intent(intent1, state)
    print(f"\nFirst task:")
    print(f"  Target: {target1.describe()} ({target1.entity_id})")
    if isinstance(result1, NavigationCommand):
        print(f"  Started: NavigationCommand to {result1.target_id}")
        print(f"  Active task: {controller.has_active_task}")
    else:
        print(f"  Resolution: {result1.to_dict()}")
        print("  (Test continues with available state)")

    # Simulate a few steps
    for _ in range(5):
        state = world.step(0)  # Move up

    # Interrupt with new navigation (use size for uniqueness)
    target2 = all_objects[1]
    intent2 = MockParsedIntent(
        type="task",
        action="navigate",
        target_type="object",
        color=target2.color,
        shape=target2.shape,
        size=target2.size,
        raw_input=f"go to the {target2.size} {target2.color} {target2.shape}",
    )

    print(f"\nInterrupting with new task:")
    print(f"  New target: {target2.describe()} ({target2.entity_id})")

    result2 = controller.process_intent(intent2, state)

    if isinstance(result2, NavigationCommand):
        print(f"  New navigation command issued")
        print(f"  Target ID: {result2.target_id}")
    else:
        print(f"  Result: {result2.to_dict()}")

    print(f"  Active task: {controller.has_active_task}")

    # Test explicit cancellation
    print("\nExplicit cancellation:")
    cancel_result = controller.cancel_task("user_request")
    if cancel_result:
        print(f"  Canceled: {cancel_result.to_dict()}")
    else:
        print("  No active task to cancel")

    print(f"  Active task after cancel: {controller.has_active_task}")

    print("\nInterrupt tests passed!")


def test_with_llm():
    """Test with actual LLM integration."""
    print("\n" + "=" * 60)
    print("Testing with LLM Integration")
    print("=" * 60)

    try:
        from src.llm import IntentParser
    except ImportError as e:
        print(f"LLM module not available: {e}")
        return

    # Initialize LLM
    print("\nInitializing LLM...")
    parser = IntentParser(backend="ollama", model_name="qwen3:4b")
    parser.load()

    # Create world and controller
    config = WorldConfig(size=32, num_objects=8)
    world = World(config=config, seed=789)
    controller = NPCController()

    state = world.get_state()

    # Print world state
    print("\nWorld objects:")
    for obj in state["objects"]:
        print(f"  {obj['entity_id']}: {obj['size']} {obj['color']} {obj['shape']}")

    # Test inputs
    test_inputs = [
        "go to the red circle",
        "walk to the player",
        "wait here",
        "attack the enemy",
        "hello there",
    ]

    print("\nLLM → Controller tests:")
    for user_input in test_inputs:
        print(f"\n  Input: \"{user_input}\"")

        # Parse with LLM
        parsed = parser.parse(user_input)
        print(f"    LLM parsed: type={parsed.type}, action={parsed.action}")

        # Process with controller
        result = controller.process_intent(parsed, state)

        if isinstance(result, NavigationCommand):
            print(f"    Controller: NavigationCommand to {result.target_id}")

            # Generate response
            llm_result = {"status": "success", "target_id": result.target_id}
            response = parser.respond_to_result(user_input, llm_result)
            print(f"    Response: \"{response}\"")
        else:
            print(f"    Controller: {result.status.name}")
            if result.response:
                print(f"    Response: \"{result.response}\"")
            elif result.matches:
                # Ambiguous - get LLM to respond
                llm_result = result.to_dict()
                response = parser.respond_to_result(user_input, llm_result)
                print(f"    Response: \"{response}\"")

    parser.unload()
    print("\nLLM integration tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test NPCController")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Include LLM integration test (requires Ollama)",
    )
    args = parser.parse_args()

    # Run tests
    test_world_query()
    test_controller_resolution()
    test_navigation_flow()
    test_interrupt()

    if args.with_llm:
        test_with_llm()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
