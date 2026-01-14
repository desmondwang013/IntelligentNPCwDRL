"""
Gym environment for training the movement policy.

This environment receives a target from the target selection policy
and trains the NPC to navigate to it. No focus_hint cheating -
the movement policy learns pure navigation.
"""
import random
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.runtime import Runtime, RuntimeConfig
from src.world.world import WorldConfig
from src.reward import RewardConfig
from src.intent import IntentType
from src.observation import TargetSelectionObsBuilder


class MovementEnv(gym.Env):
    """
    Environment for learning navigation to a specified target.

    This is designed to work with the two-policy architecture:
    1. Target selection policy picks an object (external)
    2. This environment trains movement to reach that object

    The observation does NOT include focus_hint - the movement
    policy must learn to navigate without knowing which object
    is "correct" in the embedding sense.

    Episode structure:
    - Episode starts: world resets, target index provided
    - Episode step: NPC takes movement action, gets reward
    - Episode ends: NPC reaches target, timeout, or max steps

    Action space: Discrete(5) - UP, DOWN, LEFT, RIGHT, WAIT
    Observation space: Box(567,) - same as before but NO focus_hint
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        runtime_config: Optional[RuntimeConfig] = None,
        world_config: Optional[WorldConfig] = None,
        max_steps_per_episode: int = 500,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self._runtime_config = runtime_config or RuntimeConfig()
        self._world_config = world_config
        self._max_steps = max_steps_per_episode
        self._seed = seed

        # Create runtime (will be reset in reset())
        self._runtime: Optional[Runtime] = None
        self._current_step = 0
        self._target_index: Optional[int] = None
        self._target_id: Optional[str] = None
        self._rng = random.Random(seed)

        # For converting target index to entity_id
        self._target_obs_builder = TargetSelectionObsBuilder()

        # Define spaces
        self.action_space = spaces.Discrete(5)  # Movement only

        # Observation: 575 - 8 (no focus_hint) = 567
        # Actually let's keep the same structure but zero out focus_hint
        # This maintains compatibility and makes dim calculation easier
        temp_runtime = Runtime(config=self._runtime_config, world_config=self._world_config)
        obs_dim = temp_runtime.observation_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed
            options: Must contain 'target_index' (0-9) specifying which object to navigate to.
                    If not provided, a random target is selected (for standalone testing).

        Returns:
            observation: Initial observation (no focus_hint)
            info: Additional info dict
        """
        if seed is not None:
            self._rng = random.Random(seed)
            self._seed = seed

        # Create fresh runtime
        world_seed = self._rng.randint(0, 1000000)
        self._runtime_config.world_seed = world_seed
        self._runtime = Runtime(
            config=self._runtime_config,
            world_config=self._world_config,
        )

        self._current_step = 0

        # Get target index from options, or pick random
        if options and "target_index" in options:
            self._target_index = options["target_index"]
        else:
            self._target_index = self._rng.randint(0, 9)

        # Convert index to entity_id
        world_state = self._runtime.get_state()["world"]
        self._target_id = self._target_obs_builder.get_object_id_by_index(
            world_state, self._target_index
        )

        # Get target object info for instruction generation
        target_obj = None
        for obj in world_state["objects"]:
            if obj["entity_id"] == self._target_id:
                target_obj = obj
                break

        if target_obj is None:
            raise RuntimeError(f"Could not find object at index {self._target_index}")

        # Submit instruction (text is for embedding, target_id is for completion check)
        instruction = f"Go to the {target_obj['color']} {target_obj['shape']}"
        self._runtime.submit_instruction(
            text=instruction,
            intent_type=IntentType.MOVE_TO_OBJECT,
            target_entity_id=self._target_id,
            distance_threshold=2.0,
        )

        # Process the instruction
        self._runtime.step(4)  # WAIT action to process

        # Get observation WITHOUT focus_hint
        obs = self._get_observation_with_target()

        info = {
            "target_index": self._target_index,
            "target_id": self._target_id,
            "instruction": instruction,
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action ID (0-4: UP, DOWN, LEFT, RIGHT, WAIT)

        Returns:
            observation: New observation (no focus_hint)
            reward: Reward for this step
            terminated: Whether episode ended
            truncated: Whether episode was cut short
            info: Additional info
        """
        if self._runtime is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Execute action
        result = self._runtime.step(action)
        self._current_step += 1

        # Get observation without focus_hint
        obs = self._get_observation_with_target()
        reward = result.reward

        # Check termination
        terminated = False
        truncated = False

        if not result.intent_state.get("has_intent", False):
            terminated = True

        if self._current_step >= self._max_steps:
            truncated = True

        # Build info
        info = {
            "tick": result.tick,
            "reward_info": result.reward_info.to_dict(),
            "intent_state": result.intent_state,
            "target_index": self._target_index,
            "target_id": self._target_id,
        }

        for event in result.events:
            if event.event_type.name in ("INTENT_COMPLETED", "INTENT_TIMEOUT", "INTENT_CANCELED"):
                info["termination_reason"] = event.event_type.name

        return obs, reward, terminated, truncated, info

    def _get_observation_with_target(self) -> np.ndarray:
        """
        Get observation with target encoded as focus_hint.

        The observation includes 8 nearest objects. We need to tell
        the agent which of these 8 is the target. We do this by
        setting the focus_hint one-hot to point to the correct slot.

        If the target isn't in the 8 nearest, we point to the closest
        one to the target (as a proxy).
        """
        # Get base observation (has focus_hint from runtime, but we'll override)
        obs = self._runtime.get_observation()

        # Zero out the focus_hint (last 8 elements)
        obs[-8:] = 0.0

        # Get world state to find target object position
        world_state = self._runtime.get_state()["world"]
        objects = world_state["objects"]
        npc_pos = world_state["npc"]["position"]

        # Find target object
        target_obj = None
        for obj in objects:
            if obj["entity_id"] == self._target_id:
                target_obj = obj
                break

        if target_obj is None:
            return obs  # No target, return zeros

        # Sort objects by distance to NPC (same as observation builder)
        def distance_to_npc(obj):
            ox, oy = obj["position"]["x"], obj["position"]["y"]
            nx, ny = npc_pos["x"], npc_pos["y"]
            return ((ox - nx) ** 2 + (oy - ny) ** 2) ** 0.5

        sorted_by_distance = sorted(objects, key=distance_to_npc)
        nearest_8 = sorted_by_distance[:8]

        # Find which slot (0-7) the target is in
        for i, obj in enumerate(nearest_8):
            if obj["entity_id"] == self._target_id:
                obs[-8 + i] = 1.0  # Set one-hot for this slot
                break
        else:
            # Target not in nearest 8 - find closest of the 8 to target
            target_pos = target_obj["position"]
            def distance_to_target(obj):
                ox, oy = obj["position"]["x"], obj["position"]["y"]
                tx, ty = target_pos["x"], target_pos["y"]
                return ((ox - tx) ** 2 + (oy - ty) ** 2) ** 0.5

            closest_to_target = min(range(8), key=lambda i: distance_to_target(nearest_8[i]))
            obs[-8 + closest_to_target] = 0.5  # Partial signal (not exactly the target)

        return obs

    def render(self) -> None:
        """Render the environment."""
        if self._runtime is None:
            print("Environment not initialized.")
            return

        state = self._runtime.get_state()
        npc = state["world"]["npc"]["position"]
        intent = state["intent"]

        print(f"Tick: {state['tick']}, NPC: ({npc['x']:.1f}, {npc['y']:.1f})")
        print(f"Target index: {self._target_index}, id: {self._target_id}")
        if intent.get("has_intent"):
            print(f"Intent: {intent['intent']['text']}")

    def close(self) -> None:
        """Clean up resources."""
        self._runtime = None

    @property
    def runtime(self) -> Optional[Runtime]:
        """Access the underlying runtime."""
        return self._runtime
