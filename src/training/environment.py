"""
Gym-compatible environment wrapper for the NPC Runtime.
Makes our system compatible with stable-baselines3 PPO.
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


class NPCEnv(gym.Env):
    """
    Gym environment that wraps the NPC Runtime.

    Episode structure:
    - Episode starts: world resets, random instruction given
    - Episode step: NPC takes action, gets reward
    - Episode ends: intent completes, times out, or max steps reached

    Action space: Discrete(5) - UP, DOWN, LEFT, RIGHT, WAIT
                  (SPEAK excluded for MVP simplicity)

    Observation space: Box(575,) - the observation vector from ObservationBuilder
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        runtime_config: Optional[RuntimeConfig] = None,
        world_config: Optional[WorldConfig] = None,
        max_steps_per_episode: int = 500,
        instruction_templates: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self._runtime_config = runtime_config or RuntimeConfig()
        self._world_config = world_config
        self._max_steps = max_steps_per_episode
        self._seed = seed

        # Instruction templates for generating random tasks
        self._instruction_templates = instruction_templates or [
            "Go to the {color} {shape}",
            "Move to the {color} {shape}",
            "Walk to the {color} {shape}",
        ]

        # Create runtime (will be reset in reset())
        self._runtime: Optional[Runtime] = None
        self._current_step = 0
        self._current_target_id: Optional[str] = None
        self._rng = random.Random(seed)

        # Define spaces
        # Action: 5 discrete actions (excluding SPEAK for MVP)
        self.action_space = spaces.Discrete(5)

        # Observation: 575-dim vector
        # We need to initialize runtime briefly to get observation dim
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

        Returns:
            observation: Initial observation
            info: Additional info dict
        """
        if seed is not None:
            self._rng = random.Random(seed)
            self._seed = seed

        # Create fresh runtime with random world seed
        world_seed = self._rng.randint(0, 1000000)
        self._runtime_config.world_seed = world_seed
        self._runtime = Runtime(
            config=self._runtime_config,
            world_config=self._world_config,
        )

        self._current_step = 0

        # Generate and submit a random instruction
        self._submit_random_instruction()

        # Get initial observation
        obs = self._runtime.get_observation()

        info = {
            "target_id": self._current_target_id,
            "instruction": self._runtime.intent_manager.active_intent.text if self._runtime.intent_manager.active_intent else None,
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
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (intent completed/failed)
            truncated: Whether episode was cut short (max steps)
            info: Additional info
        """
        if self._runtime is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Execute action
        result = self._runtime.step(action)
        self._current_step += 1

        # Get observation and reward
        obs = result.observation
        reward = result.reward

        # Check termination conditions
        terminated = False
        truncated = False

        # Check if intent ended (completed or timeout)
        if not result.intent_state.get("has_intent", False):
            terminated = True

        # Check max steps
        if self._current_step >= self._max_steps:
            truncated = True

        # Build info dict
        info = {
            "tick": result.tick,
            "reward_info": result.reward_info.to_dict(),
            "intent_state": result.intent_state,
            "target_id": self._current_target_id,
        }

        # Add termination reason
        for event in result.events:
            if event.event_type.name in ("INTENT_COMPLETED", "INTENT_TIMEOUT", "INTENT_CANCELED"):
                info["termination_reason"] = event.event_type.name

        return obs, reward, terminated, truncated, info

    def _submit_random_instruction(self) -> None:
        """Generate and submit a random instruction."""
        # Get world state to pick a random target object
        state = self._runtime.get_state()
        objects = state["world"]["objects"]

        # Pick a random object as target
        target = self._rng.choice(objects)
        self._current_target_id = target["entity_id"]

        # Generate instruction text
        template = self._rng.choice(self._instruction_templates)
        instruction = template.format(
            color=target["color"],
            shape=target["shape"],
        )

        # Submit instruction
        self._runtime.submit_instruction(
            text=instruction,
            intent_type=IntentType.MOVE_TO_OBJECT,
            target_entity_id=target["entity_id"],
            distance_threshold=2.0,
        )

        # Process the instruction (advances one tick)
        self._runtime.step(4)  # WAIT action to process the event

    def render(self) -> None:
        """Render the environment (text-based for now)."""
        if self._runtime is None:
            print("Environment not initialized.")
            return

        state = self._runtime.get_state()
        npc = state["world"]["npc"]["position"]
        intent = state["intent"]

        print(f"Tick: {state['tick']}, NPC: ({npc['x']:.1f}, {npc['y']:.1f})")
        if intent.get("has_intent"):
            print(f"Intent: {intent['intent']['text']}")

    def close(self) -> None:
        """Clean up resources."""
        self._runtime = None

    @property
    def runtime(self) -> Optional[Runtime]:
        """Access the underlying runtime (for debugging)."""
        return self._runtime
