"""
Simplified observation builder for RL executor.

This builder creates observations that contain ONLY structured goal information,
no language embeddings. This aligns with the architecture where:
- LLM handles language understanding
- Target resolver provides concrete target_id
- RL executor receives structured goals and executes motor skills
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class SimpleObservationConfig:
    """Configuration for simplified observation vector."""
    world_size: int = 64
    num_obstacles: int = 5  # Number of nearest obstacles to include

    @property
    def max_distance(self) -> float:
        """Max distance for normalization, scales with world size (diagonal)."""
        return self.world_size * 1.414  # sqrt(2) * world_size


# Per-obstacle features: relative position (2) + collision radius (1)
FEATURES_PER_OBSTACLE = 3


class SimpleObservationBuilder:
    """
    Builds simplified observation vectors for the RL executor.

    The RL agent receives ONLY what it needs for motor execution:
    - Where am I?
    - Where is my target?
    - What direction should I go?
    - What obstacles are nearby?

    NO language embeddings - that's the LLM's job.

    Observation structure:
    - NPC position (2): x, y normalized to [0, 1]
    - Target position (2): x, y normalized to [0, 1]
    - Direction to target (2): unit vector pointing toward target
    - Distance to target (1): normalized by max_distance
    - Target reached flag (1): 1.0 if within threshold, else 0.0
    - Nearest obstacles (5 × 3 = 15): relative position + radius
    - Intent age (1): normalized, for time pressure signal

    Total: 2 + 2 + 2 + 1 + 1 + 15 + 1 = 24 floats
    """

    def __init__(self, config: Optional[SimpleObservationConfig] = None):
        self.config = config or SimpleObservationConfig()
        self._observation_dim: Optional[int] = None

    @property
    def observation_dim(self) -> int:
        """Total dimension of the observation vector."""
        if self._observation_dim is None:
            self._observation_dim = (
                2 +  # NPC position
                2 +  # Target position
                2 +  # Direction to target
                1 +  # Distance to target
                1 +  # Target reached flag
                self.config.num_obstacles * FEATURES_PER_OBSTACLE +  # Obstacles
                1    # Intent age
            )
        return self._observation_dim

    def build(
        self,
        world_state: Dict[str, Any],
        target_id: str,
        intent_age_ticks: int = 0,
        distance_threshold: float = 2.0,
        max_intent_age_ticks: int = 960,
    ) -> np.ndarray:
        """
        Build the observation vector from world state and target information.

        Args:
            world_state: From world.get_state()
            target_id: The resolved target entity ID (e.g., "obj_3")
            intent_age_ticks: How long the current intent has been active
            distance_threshold: Distance at which target is considered "reached"
            max_intent_age_ticks: For normalizing intent age

        Returns:
            Fixed-length numpy array (24 floats)
        """
        obs_parts: List[np.ndarray] = []
        world_size = self.config.world_size

        # Extract NPC position
        npc_pos = world_state["npc"]["position"]
        npc_x, npc_y = npc_pos["x"], npc_pos["y"]

        # Find target position
        target_x, target_y = self._get_target_position(world_state, target_id)

        # 1. NPC position (normalized)
        obs_parts.append(np.array([
            npc_x / world_size,
            npc_y / world_size,
        ], dtype=np.float32))

        # 2. Target position (normalized)
        obs_parts.append(np.array([
            target_x / world_size,
            target_y / world_size,
        ], dtype=np.float32))

        # 3. Direction to target (unit vector)
        dx = target_x - npc_x
        dy = target_y - npc_y
        distance = np.sqrt(dx * dx + dy * dy)

        if distance > 0.001:  # Avoid division by zero
            direction = np.array([dx / distance, dy / distance], dtype=np.float32)
        else:
            direction = np.array([0.0, 0.0], dtype=np.float32)
        obs_parts.append(direction)

        # 4. Distance to target (normalized)
        normalized_distance = min(distance / self.config.max_distance, 1.0)
        obs_parts.append(np.array([normalized_distance], dtype=np.float32))

        # 5. Target reached flag
        reached = 1.0 if distance <= distance_threshold else 0.0
        obs_parts.append(np.array([reached], dtype=np.float32))

        # 6. Nearest obstacles (excluding target)
        obstacles = self._get_nearest_obstacles(
            world_state["objects"],
            npc_x, npc_y,
            target_id,
            self.config.num_obstacles
        )

        for obs in obstacles:
            obs_features = self._encode_obstacle(obs, npc_x, npc_y, world_size)
            obs_parts.append(obs_features)

        # Pad if fewer obstacles than expected
        num_missing = self.config.num_obstacles - len(obstacles)
        if num_missing > 0:
            # Use large distance to indicate "no obstacle here"
            zero_obstacle = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            for _ in range(num_missing):
                obs_parts.append(zero_obstacle)

        # 7. Intent age (normalized)
        normalized_age = min(intent_age_ticks / max_intent_age_ticks, 1.0)
        obs_parts.append(np.array([normalized_age], dtype=np.float32))

        # Concatenate all parts
        observation = np.concatenate(obs_parts)

        assert observation.shape[0] == self.observation_dim, \
            f"Observation dim mismatch: {observation.shape[0]} vs {self.observation_dim}"

        return observation

    def _get_target_position(
        self,
        world_state: Dict[str, Any],
        target_id: str
    ) -> tuple:
        """Get the position of the target entity."""
        # Check if target is the user
        if target_id == "user" or target_id == world_state["user"]["entity_id"]:
            pos = world_state["user"]["position"]
            return pos["x"], pos["y"]

        # Check objects
        for obj in world_state["objects"]:
            if obj["entity_id"] == target_id:
                return obj["position"]["x"], obj["position"]["y"]

        # Fallback: return center of world (shouldn't happen with valid target_id)
        return self.config.world_size / 2, self.config.world_size / 2

    def _get_nearest_obstacles(
        self,
        objects: List[Dict],
        npc_x: float,
        npc_y: float,
        target_id: str,
        n: int
    ) -> List[Dict]:
        """Get the n nearest objects (excluding the target) as obstacles."""
        def distance(obj: Dict) -> float:
            ox, oy = obj["position"]["x"], obj["position"]["y"]
            return ((ox - npc_x) ** 2 + (oy - npc_y) ** 2) ** 0.5

        # Filter out the target - we don't want to avoid it!
        obstacles = [obj for obj in objects if obj["entity_id"] != target_id]
        sorted_obstacles = sorted(obstacles, key=distance)
        return sorted_obstacles[:n]

    def _encode_obstacle(
        self,
        obj: Dict,
        npc_x: float,
        npc_y: float,
        world_size: float,
    ) -> np.ndarray:
        """
        Encode an obstacle into a feature vector.

        Features (3 total):
        - Relative position to NPC (2), normalized
        - Collision radius (1), normalized
        """
        ox, oy = obj["position"]["x"], obj["position"]["y"]

        features = [
            (ox - npc_x) / world_size,
            (oy - npc_y) / world_size,
            obj["collision_radius"] / 2.0,  # Normalize (max ~1.5)
        ]

        return np.array(features, dtype=np.float32)

    def get_observation_spec(self) -> Dict[str, Any]:
        """
        Return a specification of the observation structure.
        Useful for documentation and debugging.
        """
        n_obs = self.config.num_obstacles
        obs_features = FEATURES_PER_OBSTACLE

        return {
            "total_dim": self.observation_dim,
            "structure": {
                "npc_position": {"start": 0, "size": 2, "desc": "NPC x,y normalized"},
                "target_position": {"start": 2, "size": 2, "desc": "Target x,y normalized"},
                "direction_to_target": {"start": 4, "size": 2, "desc": "Unit vector toward target"},
                "distance_to_target": {"start": 6, "size": 1, "desc": "Normalized distance"},
                "target_reached": {"start": 7, "size": 1, "desc": "1.0 if within threshold"},
                "obstacles": {
                    "start": 8,
                    "size": n_obs * obs_features,
                    "per_obstacle": obs_features,
                    "num_obstacles": n_obs,
                    "desc": "Nearest obstacles (rel_pos + radius)",
                },
                "intent_age": {
                    "start": 8 + n_obs * obs_features,
                    "size": 1,
                    "desc": "Normalized time elapsed",
                },
            },
            "design_rationale": (
                "This observation contains ONLY structured goal information. "
                "No language embeddings - the LLM handles language understanding, "
                "and the RL executor only needs to know WHERE to go and WHAT to avoid."
            ),
        }
