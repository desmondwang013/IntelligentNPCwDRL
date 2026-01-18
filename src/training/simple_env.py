"""
Simplified Gym environment for RL executor training.

Uses SimpleObservationBuilder which provides ONLY structured goal information,
no language embeddings. This aligns with the architecture where:
- LLM handles language understanding
- Target resolver provides concrete target_id
- RL executor receives structured goals and executes motor skills
"""
import random
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.runtime import Runtime, RuntimeConfig
from src.world.world import WorldConfig
from src.observation.simple_builder import SimpleObservationBuilder, SimpleObservationConfig
from src.intent import IntentType
from src.reward import RewardConfig


class SimpleNPCEnv(gym.Env):
    """
    Simplified Gym environment for training the RL executor.

    Key differences from NPCEnv:
    - Uses SimpleObservationBuilder (24 dims instead of 621)
    - No language embeddings in observation
    - Agent receives: positions, direction to target, obstacles
    - Designed for the "RL as motor skill executor" architecture

    Episode structure:
    - Episode starts: world resets, random target assigned
    - Episode step: NPC takes action, gets reward
    - Episode ends: reaches target, times out, or max steps reached

    Action space: Discrete(5) - UP, DOWN, LEFT, RIGHT, WAIT
    Observation space: Box(24,) - structured goal information only
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        world_size: int = 64,
        max_steps_per_episode: int = 500,
        seed: Optional[int] = None,
        distance_threshold: float = 2.0,
        min_spawn_distance: Optional[float] = None,
        num_obstacles_in_obs: int = 5,
        reward_config: Optional[RewardConfig] = None,
    ):
        super().__init__()

        self._world_size = world_size
        self._distance_threshold = distance_threshold
        self._min_spawn_distance = min_spawn_distance
        self._max_steps = max_steps_per_episode
        self._seed = seed
        self._reward_config = reward_config

        # Create configs
        self._runtime_config = RuntimeConfig(reward_config=reward_config)
        self._world_config = WorldConfig(size=world_size)

        # Create simplified observation builder
        self._obs_config = SimpleObservationConfig(
            world_size=world_size,
            num_obstacles=num_obstacles_in_obs,
        )
        self._obs_builder = SimpleObservationBuilder(config=self._obs_config)

        # Runtime and state (will be reset in reset())
        self._runtime: Optional[Runtime] = None
        self._current_step = 0
        self._current_target_id: Optional[str] = None
        self._rng = random.Random(seed)

        # Action feedback tracking
        self._previous_action: Optional[int] = None
        self._action_blocked: bool = False
        self._previous_npc_pos: Optional[Tuple[float, float]] = None

        # Define spaces
        self.action_space = spaces.Discrete(5)  # UP, DOWN, LEFT, RIGHT, WAIT

        # Simplified observation space
        obs_dim = self._obs_builder.observation_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    @property
    def world_size(self) -> int:
        """Current world size."""
        return self._world_size

    @property
    def distance_threshold(self) -> float:
        """Current distance threshold for success."""
        return self._distance_threshold

    @property
    def min_spawn_distance(self) -> float:
        """Minimum distance between NPC and target at episode start."""
        if self._min_spawn_distance is not None:
            return self._min_spawn_distance
        # Default: success_radius + margin
        margin = max(2.0, self._distance_threshold * 0.5)
        return self._distance_threshold + margin

    def set_world_size(self, new_size: int) -> None:
        """Update world size for curriculum learning."""
        self._world_size = new_size
        self._world_config = WorldConfig(size=new_size)
        self._obs_config = SimpleObservationConfig(
            world_size=new_size,
            num_obstacles=self._obs_config.num_obstacles,
        )
        self._obs_builder = SimpleObservationBuilder(config=self._obs_config)

    def set_distance_threshold(self, threshold: float) -> None:
        """Update distance threshold for success (curriculum learning)."""
        self._distance_threshold = threshold

    def set_reward_config(self, config: RewardConfig) -> None:
        """Update reward config for phased training."""
        self._reward_config = config
        self._runtime_config = RuntimeConfig(reward_config=config)

    @property
    def reward_config(self) -> Optional[RewardConfig]:
        """Current reward configuration."""
        return self._reward_config

    def _get_observation(self) -> np.ndarray:
        """Build observation using SimpleObservationBuilder."""
        state = self._runtime.get_state()
        world_state = state["world"]
        intent_age = self._runtime.intent_manager.get_intent_age(self._runtime.tick)

        return self._obs_builder.build(
            world_state=world_state,
            target_id=self._current_target_id,
            intent_age_ticks=intent_age,
            distance_threshold=self._distance_threshold,
            previous_action=self._previous_action,
            action_blocked=self._action_blocked,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
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

        # Reset action feedback state
        self._previous_action = None
        self._action_blocked = False
        self._previous_npc_pos = None

        # Select random target and submit instruction
        self._submit_random_instruction()

        # Get observation using simplified builder
        obs = self._get_observation()

        info = {
            "target_id": self._current_target_id,
            "world_size": self._world_size,
            "observation_dim": self._obs_builder.observation_dim,
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self._runtime is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Get NPC position BEFORE action (for blocked detection)
        state_before = self._runtime.get_state()
        npc_pos_before = state_before["world"]["npc"]["position"]
        pos_before = (npc_pos_before["x"], npc_pos_before["y"])

        # Execute action (Runtime handles world update and reward)
        result = self._runtime.step(action)
        self._current_step += 1

        # Get NPC position AFTER action
        state_after = self._runtime.get_state()
        npc_pos_after = state_after["world"]["npc"]["position"]
        pos_after = (npc_pos_after["x"], npc_pos_after["y"])

        # Determine if action was blocked (movement action but position didn't change)
        is_movement_action = action in (0, 1, 2, 3)  # UP, DOWN, LEFT, RIGHT
        position_changed = (pos_before[0] != pos_after[0] or pos_before[1] != pos_after[1])
        self._action_blocked = is_movement_action and not position_changed
        self._previous_action = action

        # Get observation using simplified builder (now includes action feedback)
        obs = self._get_observation()
        reward = result.reward

        # Check termination conditions
        terminated = False
        truncated = False

        if not result.intent_state.get("has_intent", False):
            terminated = True

        if self._current_step >= self._max_steps:
            truncated = True

        # Build info dict
        info = {
            "tick": result.tick,
            "reward_info": result.reward_info.to_dict(),
            "intent_state": result.intent_state,
            "target_id": self._current_target_id,
        }

        for event in result.events:
            if event.event_type.name in ("INTENT_COMPLETED", "INTENT_TIMEOUT", "INTENT_CANCELED"):
                info["termination_reason"] = event.event_type.name

        return obs, reward, terminated, truncated, info

    def _submit_random_instruction(self) -> None:
        """Select a random target and submit instruction."""
        state = self._runtime.get_state()
        objects = state["world"]["objects"]
        npc_pos = state["world"]["npc"]["position"]

        # Filter objects beyond minimum spawn distance
        min_dist = self.min_spawn_distance
        valid_targets = []
        for obj in objects:
            obj_pos = obj["position"]
            dist = np.sqrt(
                (npc_pos["x"] - obj_pos["x"])**2 +
                (npc_pos["y"] - obj_pos["y"])**2
            )
            if dist >= min_dist:
                valid_targets.append(obj)

        # If no valid targets, pick the farthest one
        if not valid_targets:
            objects_with_dist = []
            for obj in objects:
                obj_pos = obj["position"]
                dist = np.sqrt(
                    (npc_pos["x"] - obj_pos["x"])**2 +
                    (npc_pos["y"] - obj_pos["y"])**2
                )
                objects_with_dist.append((obj, dist))
            objects_with_dist.sort(key=lambda x: x[1], reverse=True)
            valid_targets = [objects_with_dist[0][0]]

        # Pick random target
        target = self._rng.choice(valid_targets)
        self._current_target_id = target["entity_id"]

        # Submit instruction (text is just for logging, not used by RL)
        instruction = f"Navigate to {target['color']} {target['shape']}"

        self._runtime.submit_instruction(
            text=instruction,
            intent_type=IntentType.MOVE_TO_OBJECT,
            target_entity_id=target["entity_id"],
            distance_threshold=self._distance_threshold,
        )

        # Process the instruction
        self._runtime.step(4)  # WAIT action to process

    def render(self) -> None:
        """Render the environment (text-based)."""
        if self._runtime is None:
            print("Environment not initialized.")
            return

        state = self._runtime.get_state()
        npc = state["world"]["npc"]["position"]

        # Find target position
        target_pos = None
        for obj in state["world"]["objects"]:
            if obj["entity_id"] == self._current_target_id:
                target_pos = obj["position"]
                break

        print(f"Step {self._current_step}, NPC: ({npc['x']:.1f}, {npc['y']:.1f})")
        if target_pos:
            dist = np.sqrt((npc['x'] - target_pos['x'])**2 + (npc['y'] - target_pos['y'])**2)
            print(f"Target: ({target_pos['x']:.1f}, {target_pos['y']:.1f}), Distance: {dist:.2f}")

    def close(self) -> None:
        """Clean up resources."""
        self._runtime = None

    @property
    def runtime(self) -> Optional[Runtime]:
        """Access the underlying runtime (for debugging)."""
        return self._runtime
