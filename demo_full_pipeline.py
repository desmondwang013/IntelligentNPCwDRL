"""
Full Pipeline Demo: LLM → Controller → Navigator → Result → LLM Response

Demonstrates the complete flow in a single continuous 16x16 world:
1. User input (natural language)
2. LLM parses intent
3. Controller resolves target
4. Navigator (A* + RL) executes movement
5. Controller reports completion
6. LLM generates response

Usage:
    python demo_full_pipeline.py                    # Run test sequence
    python demo_full_pipeline.py --interactive     # Interactive mode
    python demo_full_pipeline.py --delay 0.05      # Faster animation
"""
import argparse
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.world import World
from src.world.world import WorldConfig, Action
from src.controller import NPCController, WorldQuery, NavigationCommand, TaskStatus
from src.navigation import HybridNavigator, HybridNavigatorConfig, NavigationTarget
from src.observation.simple_builder import SimpleObservationBuilder, SimpleObservationConfig
from src.llm import IntentParser


class PipelineDemo:
    """Manages the full pipeline demo."""

    def __init__(
        self,
        world_size: int = 16,
        num_objects: int = 8,
        model_path: Optional[str] = None,
        delay: float = 0.1,
        seed: int = 42,
    ):
        self.world_size = world_size
        self.delay = delay
        self.seed = seed

        # Create world
        print("Initializing world...")
        self.world_config = WorldConfig(
            size=world_size,
            num_objects=num_objects,
        )
        self.world = World(config=self.world_config, seed=seed)

        # Create controller
        print("Initializing controller...")
        self.controller = NPCController(
            default_timeout_ticks=480,
            arrival_threshold=2.0,
        )

        # Create navigator
        print("Initializing navigator...")
        self.navigator = HybridNavigator(HybridNavigatorConfig(
            waypoint_threshold=1.0,
            goal_threshold=2.0,
            grid_resolution=0.5,
            obstacle_padding=0.15,
            smooth_path=True,
            replan_on_stuck=True,
        ))

        # Create observation builder
        self.obs_builder = SimpleObservationBuilder(SimpleObservationConfig(
            world_size=world_size,
            num_obstacles=5,
        ))

        # Load RL model
        print("Loading RL model...")
        self.model = None
        if model_path is None:
            candidates = [
                "models/simple/best/best_model.zip",
                "models/simple/final_model.zip",
            ]
            for path in candidates:
                if Path(path).exists():
                    model_path = path
                    break

        if model_path and Path(path).exists():
            self.model = PPO.load(model_path)
            print(f"  Loaded: {model_path}")
        else:
            print("  No model found, using greedy policy")

        # Initialize LLM
        print("Initializing LLM...")
        self.llm = IntentParser(backend="ollama", model_name="qwen3:4b")
        self.llm.load()
        print("  LLM ready")

        # State tracking
        self.total_steps = 0
        self.command_history: List[dict] = []

    def print_world_state(self, highlight_target: Optional[str] = None):
        """Print ASCII visualization of the world."""
        state = self.world.get_state()
        world_size = state["world_size"]
        npc_pos = state["npc"]["position"]
        user_pos = state["user"]["position"]
        objects = state["objects"]

        # Grid setup
        grid_size = min(16, world_size)
        scale = world_size / grid_size
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

        # Place waypoints if navigator has them
        for wx, wy in self.navigator.waypoints:
            gx = int(wx / scale)
            gy = int(wy / scale)
            gx = max(0, min(grid_size - 1, gx))
            gy = max(0, min(grid_size - 1, gy))
            grid[grid_size - 1 - gy][gx] = "*"

        # Place objects
        for obj in objects:
            gx = int(obj["position"]["x"] / scale)
            gy = int(obj["position"]["y"] / scale)
            gx = max(0, min(grid_size - 1, gx))
            gy = max(0, min(grid_size - 1, gy))

            if obj["entity_id"] == highlight_target:
                grid[grid_size - 1 - gy][gx] = "T"  # Target
            else:
                # Use first letter of color
                grid[grid_size - 1 - gy][gx] = obj["color"][0].upper()

        # Place user
        ux = int(user_pos["x"] / scale)
        uy = int(user_pos["y"] / scale)
        ux = max(0, min(grid_size - 1, ux))
        uy = max(0, min(grid_size - 1, uy))
        grid[grid_size - 1 - uy][ux] = "U"

        # Place NPC (overwrites)
        nx = int(npc_pos["x"] / scale)
        ny = int(npc_pos["y"] / scale)
        nx = max(0, min(grid_size - 1, nx))
        ny = max(0, min(grid_size - 1, ny))
        grid[grid_size - 1 - ny][nx] = "@"

        # Print
        print("+" + "-" * grid_size + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * grid_size + "+")
        print("Legend: @=NPC, U=User, T=Target, R/B/G/Y/P/O=objects, *=path")

    def print_objects_list(self):
        """Print list of objects in the world."""
        state = self.world.get_state()
        print("\nObjects in world:")
        for obj in state["objects"]:
            pos = obj["position"]
            print(f"  {obj['entity_id']}: {obj['size']} {obj['color']} {obj['shape']} at ({pos['x']:.1f}, {pos['y']:.1f})")

    def run_navigation(self, command: NavigationCommand) -> dict:
        """Execute navigation using RL model."""
        state = self.world.get_state()
        npc_pos = state["npc"]["position"]

        # Set up navigator
        path_found = self.navigator.set_goal(
            npc_pos["x"], npc_pos["y"],
            command.target_position[0], command.target_position[1],
            state,
        )

        if not path_found:
            return {"success": False, "reason": "no path found", "steps": 0}

        print(f"\n  Path planned: {len(self.navigator.waypoints)} waypoints")

        # Tracking
        steps = 0
        max_steps = 300
        previous_action = 4  # WAIT
        action_blocked = False

        while steps < max_steps:
            current_waypoint = self.navigator.get_current_target()
            if current_waypoint is None:
                break

            # Get current state
            state = self.world.get_state()
            npc_pos = state["npc"]["position"]

            # Build observation
            obs = self.obs_builder.build(
                world_state=state,
                target_position=(current_waypoint.x, current_waypoint.y),
                intent_age_ticks=steps,
                distance_threshold=2.0,
                previous_action=previous_action,
                action_blocked=action_blocked,
            )

            # Get action
            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = self._greedy_action(
                    npc_pos["x"], npc_pos["y"],
                    current_waypoint.x, current_waypoint.y
                )

            # Record position before action
            pos_before = (npc_pos["x"], npc_pos["y"])

            # Take step
            state = self.world.step(action)
            steps += 1
            self.total_steps += 1

            # Check if blocked
            npc_pos = state["npc"]["position"]
            pos_after = (npc_pos["x"], npc_pos["y"])
            action_blocked = (
                action in [0, 1, 2, 3] and
                pos_before[0] == pos_after[0] and
                pos_before[1] == pos_after[1]
            )
            previous_action = action

            # Update navigator
            self.navigator.update(npc_pos["x"], npc_pos["y"], state)

            # Check controller timeout
            timeout_result = self.controller.update(state)
            if timeout_result:
                return {"success": False, "reason": "timeout", "steps": steps}

            # Visualization (every N steps or final)
            nav_state = self.navigator.get_state()
            if steps % 10 == 0 or nav_state.is_complete:
                print(f"\r  Step {steps}: pos=({npc_pos['x']:.1f}, {npc_pos['y']:.1f}), "
                      f"waypoints remaining={nav_state.waypoints_remaining}", end="")

            # Check completion
            if nav_state.is_complete:
                print()  # Newline after progress
                return {"success": True, "steps": steps}

            if nav_state.is_stuck:
                print()
                return {"success": False, "reason": "stuck", "steps": steps}

            time.sleep(self.delay)

        print()
        return {"success": False, "reason": "max steps", "steps": steps}

    def _greedy_action(self, npc_x: float, npc_y: float, target_x: float, target_y: float) -> int:
        """Fallback greedy action."""
        dx = target_x - npc_x
        dy = target_y - npc_y
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        elif abs(dy) > 0.1:
            return 0 if dy > 0 else 1
        return 4

    def process_input(self, user_input: str) -> str:
        """Process a single user input through the full pipeline."""
        print("\n" + "=" * 60)
        print(f"USER: \"{user_input}\"")
        print("=" * 60)

        state = self.world.get_state()

        # Step 1: LLM parses intent
        print("\n[1] LLM Parsing...")
        start_time = time.time()
        parsed = self.llm.parse(user_input, debug=False)
        parse_time = time.time() - start_time

        print(f"  Type: {parsed.type}")
        if parsed.type == "task":
            print(f"  Action: {parsed.action}")
            if parsed.target_type:
                print(f"  Target: {parsed.target_type}", end="")
                if parsed.color:
                    print(f", color={parsed.color}", end="")
                if parsed.shape:
                    print(f", shape={parsed.shape}", end="")
                if parsed.size:
                    print(f", size={parsed.size}", end="")
                print()
        else:
            print(f"  Response: \"{parsed.response}\"")
        print(f"  Time: {parse_time:.2f}s")

        # Step 2: Controller processes intent
        print("\n[2] Controller Processing...")
        result = self.controller.process_intent(parsed, state)

        if isinstance(result, NavigationCommand):
            print(f"  Result: NavigationCommand")
            print(f"  Target ID: {result.target_id}")
            print(f"  Position: ({result.target_position[0]:.1f}, {result.target_position[1]:.1f})")

            # Show world before navigation
            print("\n[3] Starting Navigation...")
            self.print_world_state(highlight_target=result.target_id)

            # Step 3: Execute navigation
            nav_result = self.run_navigation(result)

            # Step 4: Report completion to controller
            state = self.world.get_state()
            completion = self.controller.report_navigation_complete(
                state,
                success=nav_result["success"],
                reason=nav_result.get("reason"),
            )

            print(f"\n[4] Navigation Complete:")
            print(f"  Success: {nav_result['success']}")
            print(f"  Steps: {nav_result['steps']}")
            if not nav_result["success"]:
                print(f"  Reason: {nav_result.get('reason', 'unknown')}")

            # Show world after navigation
            print("\n  Final position:")
            self.print_world_state(highlight_target=result.target_id)

            # Step 5: LLM generates response
            print("\n[5] LLM Response Generation...")
            llm_result = completion.to_dict()
            response = self.llm.respond_to_result(user_input, llm_result)

            # Fallback if LLM response is generic
            if response in ["I'm not sure what happened.", ""]:
                if nav_result["success"]:
                    response = f"Done! I've reached the target in {nav_result['steps']} steps."
                else:
                    response = f"Sorry, I couldn't complete that. {nav_result.get('reason', 'Something went wrong')}."

            # Record history
            self.command_history.append({
                "input": user_input,
                "parsed": {"type": parsed.type, "action": parsed.action},
                "result": llm_result,
                "response": response,
            })

            print(f"\nNPC: \"{response}\"")
            return response

        else:
            # Immediate result (conversation, wait, error)
            print(f"  Result: {result.status.name}")
            result_dict = result.to_dict()
            for key, value in result_dict.items():
                if key != "status" and value is not None:
                    print(f"  {key}: {value}")

            if result.status == TaskStatus.SUCCESS and parsed.type == "conversation":
                # Conversation - response already in parsed intent
                response = parsed.response or "I'm not sure what to say."
                self.command_history.append({
                    "input": user_input,
                    "parsed": {"type": "conversation"},
                    "result": result_dict,
                    "response": response,
                })
                print(f"\nNPC: \"{response}\"")
                return response

            elif result.status == TaskStatus.AMBIGUOUS:
                # Need clarification - always include matches
                matches = result.matches or []
                if matches:
                    matches_str = ", ".join(matches)
                    response = f"Which one do you mean? I see: {matches_str}."
                else:
                    response = "I found multiple matches. Can you be more specific?"

                self.command_history.append({
                    "input": user_input,
                    "parsed": {"type": parsed.type, "action": parsed.action},
                    "result": result_dict,
                    "response": response,
                })
                print(f"\nNPC: \"{response}\"")
                return response

            elif result.status == TaskStatus.SUCCESS and parsed.action == "wait":
                # Wait command
                response = "Alright, I'll wait here."
                self.command_history.append({
                    "input": user_input,
                    "parsed": {"type": "task", "action": "wait"},
                    "result": result_dict,
                    "response": response,
                })
                print(f"\nNPC: \"{response}\"")
                return response

            elif result.status == TaskStatus.UNSUPPORTED:
                # Unsupported action
                response = self.llm.respond_to_result(user_input, result_dict)

                # Fallback if LLM response is generic
                if response in ["I'm not sure what happened.", ""]:
                    response = "Sorry, I can't do that. I can only navigate to places or wait."

                self.command_history.append({
                    "input": user_input,
                    "parsed": {"type": parsed.type, "action": parsed.action},
                    "result": result_dict,
                    "response": response,
                })
                print(f"\nNPC: \"{response}\"")
                return response

            elif result.status == TaskStatus.NOT_FOUND:
                # Target not found
                response = self.llm.respond_to_result(user_input, result_dict)

                # Fallback if LLM response is generic
                if response in ["I'm not sure what happened.", ""]:
                    response = "I don't see anything matching that description here."

                self.command_history.append({
                    "input": user_input,
                    "parsed": {"type": parsed.type, "action": parsed.action},
                    "result": result_dict,
                    "response": response,
                })
                print(f"\nNPC: \"{response}\"")
                return response

            else:
                # Other cases
                response = self.llm.respond_to_result(user_input, result_dict)

                # Generic fallback
                if response in ["I'm not sure what happened.", ""]:
                    response = "Something unexpected happened. Let me know what you'd like me to do."

                self.command_history.append({
                    "input": user_input,
                    "parsed": {"type": parsed.type, "action": parsed.action},
                    "result": result_dict,
                    "response": response,
                })
                print(f"\nNPC: \"{response}\"")
                return response

    def run_test_sequence(self, inputs: List[str]):
        """Run a sequence of test inputs."""
        print("\n" + "#" * 70)
        print("# FULL PIPELINE DEMO - Test Sequence")
        print("#" * 70)

        # Show initial world
        print("\nInitial World State:")
        self.print_objects_list()
        print()
        self.print_world_state()

        input("\nPress Enter to start test sequence...")

        for i, user_input in enumerate(inputs):
            print(f"\n\n{'#' * 70}")
            print(f"# TEST {i+1}/{len(inputs)}")
            print(f"{'#' * 70}")

            self.process_input(user_input)

            if i < len(inputs) - 1:
                input("\nPress Enter for next command...")

        # Summary
        self.print_summary()

    def run_interactive(self):
        """Run in interactive mode."""
        print("\n" + "#" * 70)
        print("# FULL PIPELINE DEMO - Interactive Mode")
        print("#" * 70)

        # Show initial world
        print("\nInitial World State:")
        self.print_objects_list()
        print()
        self.print_world_state()

        print("\nType commands for the NPC. Type 'quit' to exit, 'show' to see world.")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == "quit":
                    break
                if user_input.lower() == "show":
                    self.print_objects_list()
                    self.print_world_state()
                    continue
                if user_input.lower() == "history":
                    self.print_summary()
                    continue

                self.process_input(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except EOFError:
                break

        self.print_summary()

    def print_summary(self):
        """Print session summary."""
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"Total commands: {len(self.command_history)}")
        print(f"Total world steps: {self.total_steps}")

        if self.command_history:
            print("\nCommand history:")
            for i, cmd in enumerate(self.command_history):
                status = cmd["result"].get("status", "unknown")
                print(f"  {i+1}. \"{cmd['input']}\" -> {status}")

    def cleanup(self):
        """Clean up resources."""
        self.llm.unload()


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Demo")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--world-size", type=int, default=16,
                        help="World size (default: 16)")
    parser.add_argument("--num-objects", type=int, default=8,
                        help="Number of objects (default: 8)")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Delay between navigation steps (default: 0.02)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to RL model")

    args = parser.parse_args()

    # Test inputs for demo sequence
    test_inputs = [
        # Greetings
        "hello there",

        # Navigation to specific object (unique - should work)
        "go to the large purple circle",

        # Navigation to user
        "walk to the player",

        # Ambiguous target (multiple red objects in seed=42 world)
        "go to the red thing",

        # Wait command
        "wait here",

        # Unsupported action
        "attack the enemy",

        # Question/conversation
        "how are you today?",
    ]

    # Create demo
    demo = PipelineDemo(
        world_size=args.world_size,
        num_objects=args.num_objects,
        model_path=args.model,
        delay=args.delay,
        seed=args.seed,
    )

    try:
        if args.interactive:
            demo.run_interactive()
        else:
            demo.run_test_sequence(test_inputs)
    finally:
        demo.cleanup()

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
