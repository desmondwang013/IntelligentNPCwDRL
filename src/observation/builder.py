"""
LEGACY: This observation builder includes language embeddings (384 dimensions).

In the current architecture, RL receives only structured goals, no language.
Use `SimpleObservationBuilder` from `src/observation/simple_builder.py` instead.

This file is kept for backward compatibility with:
- Legacy NPCEnv
- Old training scripts (train.py, train_curriculum*.py)
- Demo scripts that still use embeddings

See README.md for the current architecture: LLM → Target Resolver → RL Executor
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from src.world.entities import ObjectType, ObjectColor, ObjectShape, ObjectSize


@dataclass
class ObservationConfig:
    """Configuration for observation vector construction."""
    world_size: int = 64
    num_nearest_objects: int = 10  # Changed from 8 to see ALL objects (omniscient)
    embedding_dim: int = 384
    max_intent_age_ticks: int = 960  # 60 seconds for normalization


# Enum sizes for one-hot encoding
NUM_OBJECT_TYPES = len(ObjectType)    # 3: fixed, light, heavy
NUM_COLORS = len(ObjectColor)          # 6
NUM_SHAPES = len(ObjectShape)          # 4
NUM_SIZES = len(ObjectSize)            # 3

# Per-object feature size
# 2 (rel to NPC) + 2 (rel to User) + 3 (type) + 6 (color) + 4 (shape) + 3 (size) + 1 (movable) + 1 (radius)
FEATURES_PER_OBJECT = 2 + 2 + NUM_OBJECT_TYPES + NUM_COLORS + NUM_SHAPES + NUM_SIZES + 1 + 1  # = 22


class ObservationBuilder:
    """
    Builds fixed-length observation vectors for the policy.

    Observation structure:
    - NPC position (2): x, y normalized to [0, 1]
    - User position (2): x, y normalized to [0, 1]
    - NPC→User offset (2): relative position, normalized
    - Nearest objects (8 × 22 = 176): see _encode_object
    - Intent embedding (384): from sentence transformer
    - Intent age (1): normalized by max age
    - Focus hint (8): one-hot indicating which of the 8 nearest objects is focused

    Total: 2 + 2 + 2 + 176 + 384 + 1 + 8 = 575 floats
    """

    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()
        self._observation_dim: Optional[int] = None

    @property
    def observation_dim(self) -> int:
        """Total dimension of the observation vector."""
        if self._observation_dim is None:
            self._observation_dim = (
                2 +  # NPC position
                2 +  # User position
                2 +  # NPC→User offset
                self.config.num_nearest_objects * FEATURES_PER_OBJECT +  # Objects
                self.config.embedding_dim +  # Intent embedding
                1 +  # Intent age
                self.config.num_nearest_objects  # Focus hint one-hot
            )
        return self._observation_dim

    def build(
        self,
        world_state: Dict[str, Any],
        intent_embedding: np.ndarray,
        intent_age_ticks: int,
        focus_hint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build the full observation vector from world state and intent info.

        Args:
            world_state: From world.get_state()
            intent_embedding: 384-dim vector from IntentManager
            intent_age_ticks: How long the current intent has been active
            focus_hint: Optional object ID that the intent likely refers to

        Returns:
            Fixed-length numpy array (575 floats)
        """
        obs_parts: List[np.ndarray] = []
        world_size = self.config.world_size

        # Extract positions
        npc_pos = world_state["npc"]["position"]
        user_pos = world_state["user"]["position"]
        npc_x, npc_y = npc_pos["x"], npc_pos["y"]
        user_x, user_y = user_pos["x"], user_pos["y"]

        # 1. NPC position (normalized)
        obs_parts.append(np.array([
            npc_x / world_size,
            npc_y / world_size,
        ], dtype=np.float32))

        # 2. User position (normalized)
        obs_parts.append(np.array([
            user_x / world_size,
            user_y / world_size,
        ], dtype=np.float32))

        # 3. NPC→User offset (normalized)
        obs_parts.append(np.array([
            (user_x - npc_x) / world_size,
            (user_y - npc_y) / world_size,
        ], dtype=np.float32))

        # 4. Nearest objects
        nearest_objects = self._get_nearest_objects(
            world_state["objects"],
            npc_x, npc_y,
            self.config.num_nearest_objects
        )

        # Track focus hint index for later
        focus_index = -1

        for i, obj in enumerate(nearest_objects):
            obj_features = self._encode_object(
                obj, npc_x, npc_y, user_x, user_y, world_size
            )
            obs_parts.append(obj_features)

            # Check if this object matches focus hint
            if focus_hint is not None and obj["entity_id"] == focus_hint:
                focus_index = i

        # Pad if fewer than num_nearest_objects
        num_missing = self.config.num_nearest_objects - len(nearest_objects)
        if num_missing > 0:
            zero_obj = np.zeros(FEATURES_PER_OBJECT, dtype=np.float32)
            for _ in range(num_missing):
                obs_parts.append(zero_obj)

        # 5. Intent embedding
        obs_parts.append(intent_embedding.astype(np.float32))

        # 6. Intent age (normalized)
        normalized_age = min(intent_age_ticks / self.config.max_intent_age_ticks, 1.0)
        obs_parts.append(np.array([normalized_age], dtype=np.float32))

        # 7. Focus hint (one-hot over nearest objects)
        focus_one_hot = np.zeros(self.config.num_nearest_objects, dtype=np.float32)
        if focus_index >= 0:
            focus_one_hot[focus_index] = 1.0
        obs_parts.append(focus_one_hot)

        # Concatenate all parts
        observation = np.concatenate(obs_parts)

        assert observation.shape[0] == self.observation_dim, \
            f"Observation dim mismatch: {observation.shape[0]} vs {self.observation_dim}"

        return observation

    def _get_nearest_objects(
        self,
        objects: List[Dict],
        npc_x: float,
        npc_y: float,
        n: int
    ) -> List[Dict]:
        """Get the n nearest objects to the NPC."""
        def distance(obj: Dict) -> float:
            ox, oy = obj["position"]["x"], obj["position"]["y"]
            return ((ox - npc_x) ** 2 + (oy - npc_y) ** 2) ** 0.5

        sorted_objects = sorted(objects, key=distance)
        return sorted_objects[:n]

    def _encode_object(
        self,
        obj: Dict,
        npc_x: float,
        npc_y: float,
        user_x: float,
        user_y: float,
        world_size: float,
    ) -> np.ndarray:
        """
        Encode a single object into a fixed-length feature vector.

        Features (22 total):
        - Relative position to NPC (2)
        - Relative position to User (2)
        - Object type one-hot (3)
        - Color one-hot (6)
        - Shape one-hot (4)
        - Size one-hot (3)
        - Is movable (1)
        - Collision radius normalized (1)
        """
        ox, oy = obj["position"]["x"], obj["position"]["y"]

        features: List[float] = []

        # Relative position to NPC (normalized)
        features.append((ox - npc_x) / world_size)
        features.append((oy - npc_y) / world_size)

        # Relative position to User (normalized)
        features.append((ox - user_x) / world_size)
        features.append((oy - user_y) / world_size)

        # Object type one-hot
        type_one_hot = self._one_hot(obj["object_type"], ["fixed", "light", "heavy"])
        features.extend(type_one_hot)

        # Color one-hot
        color_one_hot = self._one_hot(obj["color"], [
            "red", "blue", "green", "yellow", "purple", "orange"
        ])
        features.extend(color_one_hot)

        # Shape one-hot
        shape_one_hot = self._one_hot(obj["shape"], [
            "circle", "triangle", "square", "diamond"
        ])
        features.extend(shape_one_hot)

        # Size one-hot
        size_one_hot = self._one_hot(obj["size"], ["small", "medium", "large"])
        features.extend(size_one_hot)

        # Is movable
        features.append(1.0 if obj["is_movable"] else 0.0)

        # Collision radius (normalized, max ~1.5)
        features.append(obj["collision_radius"] / 2.0)

        return np.array(features, dtype=np.float32)

    def _one_hot(self, value: str, categories: List[str]) -> List[float]:
        """Create a one-hot encoding for a categorical value."""
        result = [0.0] * len(categories)
        if value in categories:
            result[categories.index(value)] = 1.0
        return result

    def get_observation_spec(self) -> Dict[str, Any]:
        """
        Return a specification of the observation structure.
        Useful for documentation and Unity integration.
        """
        obj_features = FEATURES_PER_OBJECT
        n_obj = self.config.num_nearest_objects

        return {
            "total_dim": self.observation_dim,
            "structure": {
                "npc_position": {"start": 0, "size": 2},
                "user_position": {"start": 2, "size": 2},
                "npc_user_offset": {"start": 4, "size": 2},
                "objects": {
                    "start": 6,
                    "size": n_obj * obj_features,
                    "per_object": obj_features,
                    "num_objects": n_obj,
                },
                "intent_embedding": {
                    "start": 6 + n_obj * obj_features,
                    "size": self.config.embedding_dim,
                },
                "intent_age": {
                    "start": 6 + n_obj * obj_features + self.config.embedding_dim,
                    "size": 1,
                },
                "focus_hint": {
                    "start": 6 + n_obj * obj_features + self.config.embedding_dim + 1,
                    "size": n_obj,
                },
            },
            "object_feature_breakdown": {
                "rel_pos_npc": 2,
                "rel_pos_user": 2,
                "type_one_hot": NUM_OBJECT_TYPES,
                "color_one_hot": NUM_COLORS,
                "shape_one_hot": NUM_SHAPES,
                "size_one_hot": NUM_SIZES,
                "is_movable": 1,
                "collision_radius": 1,
            },
        }
