"""Combat reward system for RL training."""

from dataclasses import dataclass
from typing import Dict

from src.world import World
from src.world.entities import CombatStyle


@dataclass
class CombatRewardConfig:
    """Configuration for combat rewards."""
    # Damage rewards
    damage_dealt_reward: float = 1.0  # Per point of damage dealt
    kill_bonus: float = 5.0  # Bonus for killing an enemy

    # Damage taken penalties (base values, scaled by style)
    damage_taken_penalty: float = -1.0  # Per point of damage taken

    # Style scaling for damage_taken_penalty
    # Lower multiplier = less penalty = more aggressive behavior encouraged
    aggressive_damage_scale: float = 0.3  # Barely punished for taking damage
    balanced_damage_scale: float = 1.0    # Normal penalty
    passive_damage_scale: float = 2.0     # Heavily punished for taking damage

    # Victory/defeat
    all_enemies_killed_bonus: float = 10.0
    npc_death_penalty: float = -20.0

    # Per-step cost (encourages efficiency)
    step_cost: float = -0.01

    # Curriculum phase (affects penalty scaling)
    # Phase 1: exploration (reduced penalties, attack attempt bonus)
    # Phase 2+: full penalties, no attack bonus
    phase: int = 1
    phase1_penalty_scale: float = 0.5  # Reduce penalties in phase 1
    phase1_attack_attempt_bonus: float = 0.1  # Bonus for trying attack in phase 1


class CombatReward:
    """
    Calculates rewards for combat RL training.

    Reward structure:
    - Positive: damage dealt, kills, winning
    - Negative: damage taken (scaled by combat style), death, time
    """

    def __init__(self, config: CombatRewardConfig = None):
        self.config = config or CombatRewardConfig()

    def _get_style_scale(self, style: CombatStyle) -> float:
        """Get damage penalty scale for combat style."""
        if style == CombatStyle.AGGRESSIVE:
            return self.config.aggressive_damage_scale
        elif style == CombatStyle.PASSIVE:
            return self.config.passive_damage_scale
        else:  # BALANCED
            return self.config.balanced_damage_scale

    def _get_penalty_scale(self) -> float:
        """Get penalty scale based on curriculum phase."""
        if self.config.phase == 1:
            return self.config.phase1_penalty_scale
        return 1.0

    def calculate(self, world: World, prev_alive_count: int) -> Dict[str, float]:
        """
        Calculate reward for the current step.

        Args:
            world: Current world state (after step)
            prev_alive_count: Number of alive enemies before this step

        Returns:
            Dict with 'total' reward and component breakdown
        """
        rewards = {}
        total = 0.0

        npc = world.npc
        style_scale = self._get_style_scale(npc.combat_style)
        penalty_scale = self._get_penalty_scale()

        # Damage dealt reward
        if world.last_damage_dealt > 0:
            damage_reward = world.last_damage_dealt * self.config.damage_dealt_reward
            rewards['damage_dealt'] = damage_reward
            total += damage_reward

        # Kill bonus
        if world.last_enemy_killed:
            rewards['kill'] = self.config.kill_bonus
            total += self.config.kill_bonus

        # Damage taken penalty (scaled by style and phase)
        if world.last_damage_taken > 0:
            base_penalty = world.last_damage_taken * self.config.damage_taken_penalty
            scaled_penalty = base_penalty * style_scale * penalty_scale
            rewards['damage_taken'] = scaled_penalty
            total += scaled_penalty

        # All enemies killed bonus
        current_alive = len(world.get_alive_enemies())
        if prev_alive_count > 0 and current_alive == 0:
            rewards['victory'] = self.config.all_enemies_killed_bonus
            total += self.config.all_enemies_killed_bonus

        # NPC death penalty
        if not npc.is_alive:
            death_penalty = self.config.npc_death_penalty * penalty_scale
            rewards['death'] = death_penalty
            total += death_penalty

        # Step cost
        rewards['step_cost'] = self.config.step_cost
        total += self.config.step_cost

        rewards['total'] = total
        return rewards

    def set_phase(self, phase: int) -> None:
        """Update curriculum phase."""
        self.config.phase = phase
