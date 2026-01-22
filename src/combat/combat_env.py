"""Combat training environment for RL."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from src.world import World, WorldConfig, Action
from src.world.entities import CombatStyle
from .observation import CombatObservation, CombatObservationConfig
from .reward import CombatReward, CombatRewardConfig


class CombatEnv(gym.Env):
    """
    Gymnasium environment for training combat RL agent.

    The agent controls an NPC fighting enemies. Actions are movement
    directions, wait, and attack. The goal is to defeat all enemies
    while minimizing damage taken (scaled by combat style).

    Action Space:
        Discrete(6):
        0 = MOVE_UP
        1 = MOVE_DOWN
        2 = MOVE_LEFT
        3 = MOVE_RIGHT
        4 = WAIT
        5 = ATTACK

    Observation Space:
        Box of shape (obs_size,) containing:
        - NPC state (position, health, cooldown)
        - Enemy info (relative positions, health, aggro)
        - Combat style (one-hot)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        world_config: Optional[WorldConfig] = None,
        num_enemies: int = 1,
        combat_style: CombatStyle = CombatStyle.BALANCED,
        max_steps: int = 500,
        reward_config: Optional[CombatRewardConfig] = None,
        obs_config: Optional[CombatObservationConfig] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.world_config = world_config or WorldConfig(size=16, num_objects=0)
        self.num_enemies = num_enemies
        self.combat_style = combat_style
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._seed = seed

        # Initialize components
        self.world = World(self.world_config, seed=seed)
        self.obs_builder = CombatObservation(obs_config or CombatObservationConfig(
            world_size=self.world_config.size
        ))
        self.reward_calculator = CombatReward(reward_config)

        # Spaces
        self.action_space = spaces.Discrete(6)  # 4 moves + wait + attack
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_builder.observation_size,),
            dtype=np.float32
        )

        # Episode tracking
        self.step_count = 0
        self.prev_alive_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        # Reset world with enemies
        self.world.reset(seed=self._seed, num_enemies=self.num_enemies)

        # Set combat style
        self.world.npc.combat_style = self.combat_style

        # Reset tracking
        self.step_count = 0
        self.prev_alive_count = len(self.world.get_alive_enemies())

        obs = self.obs_builder.build(self.world)
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return results."""
        self.step_count += 1

        # Map action to world action
        world_action = Action(action)

        # Store state before step
        prev_alive = len(self.world.get_alive_enemies())

        # Execute action
        self.world.step(world_action)

        # Calculate reward
        reward_info = self.reward_calculator.calculate(self.world, prev_alive)
        reward = reward_info['total']

        # Check termination
        terminated = False
        truncated = False

        # Win: all enemies dead
        if len(self.world.get_alive_enemies()) == 0:
            terminated = True

        # Lose: NPC dead
        if not self.world.npc.is_alive:
            terminated = True

        # Time limit
        if self.step_count >= self.max_steps:
            truncated = True

        # Update tracking
        self.prev_alive_count = len(self.world.get_alive_enemies())

        # Build observation
        obs = self.obs_builder.build(self.world)
        info = self._get_info()
        info['reward_breakdown'] = reward_info

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Get episode info dict."""
        npc = self.world.npc
        alive_enemies = self.world.get_alive_enemies()

        return {
            'step': self.step_count,
            'npc_health': npc.health,
            'npc_health_ratio': npc.health_ratio,
            'npc_alive': npc.is_alive,
            'enemies_alive': len(alive_enemies),
            'enemies_total': len(self.world.enemies),
            'combat_style': npc.combat_style.name,
        }

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
        return None

    def _render_ansi(self) -> str:
        """Render as ASCII string."""
        lines = []
        npc = self.world.npc
        enemies = self.world.enemies

        lines.append(f"Step: {self.step_count}/{self.max_steps}")
        lines.append(f"NPC: HP {npc.health}/{npc.max_health} | Style: {npc.combat_style.name}")
        lines.append(f"     Pos: ({npc.position.x:.1f}, {npc.position.y:.1f})")
        lines.append(f"     Can Attack: {npc.can_attack} | Cooldown: {npc.current_cooldown}")
        lines.append("")

        for i, enemy in enumerate(enemies):
            dist = npc.position.distance_to(enemy.position)
            status = "ALIVE" if enemy.is_alive else "DEAD"
            aggro = "AGGRO" if enemy.is_aggro else "PASSIVE"
            lines.append(f"Enemy {i}: HP {enemy.health}/{enemy.max_health} | {status} | {aggro}")
            lines.append(f"         Pos: ({enemy.position.x:.1f}, {enemy.position.y:.1f}) | Dist: {dist:.2f}")

        return "\n".join(lines)

    def set_combat_style(self, style: CombatStyle) -> None:
        """Change combat style for next reset."""
        self.combat_style = style

    def set_num_enemies(self, num: int) -> None:
        """Change number of enemies for next reset."""
        self.num_enemies = num

    def set_reward_phase(self, phase: int) -> None:
        """Update reward curriculum phase."""
        self.reward_calculator.set_phase(phase)
