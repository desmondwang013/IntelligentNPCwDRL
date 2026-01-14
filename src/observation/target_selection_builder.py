"""
Observation builder for target selection policy.

This builds an observation for deciding WHICH object to target,
not for navigation. Key differences from the movement observation:
1. All 10 objects included (not just 8 nearest)
2. Objects in consistent order (by entity_id: slot 0 = obj_0, etc.)
3. No focus_hint (that's what we want the agent to learn)
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from src.world.entities import ObjectType, ObjectColor, ObjectShape, ObjectSize


@dataclass
class TargetSelectionObsConfig:
    """Configuration for target selection observation."""
    world_size: int = 64
    num_objects: int = 10
    embedding_dim: int = 384


# Enum sizes for one-hot encoding
NUM_OBJECT_TYPES = len(ObjectType)    # 3: fixed, light, heavy
NUM_COLORS = len(ObjectColor)          # 6
NUM_SHAPES = len(ObjectShape)          # 4
NUM_SIZES = len(ObjectSize)            # 3

# Per-object feature size for target selection
# Position (2) + type (3) + color (6) + shape (4) + size (3) = 18
# We skip relative-to-user since it's less relevant for "which object matches the instruction"
FEATURES_PER_OBJECT_TARGET = 2 + NUM_OBJECT_TYPES + NUM_COLORS + NUM_SHAPES + NUM_SIZES  # = 18


class TargetSelectionObsBuilder:
    """
    Builds observation vectors for target selection policy.

    The goal: given an instruction embedding, which of the 10 objects
    does it refer to?

    Observation structure:
    - Instruction embedding (384): what the user said
    - All objects (10 × 18 = 180): features of each object in fixed order

    Total: 384 + 180 = 564 floats
    """

    def __init__(self, config: Optional[TargetSelectionObsConfig] = None):
        self.config = config or TargetSelectionObsConfig()
        self._observation_dim: Optional[int] = None

    @property
    def observation_dim(self) -> int:
        """Total dimension of the observation vector."""
        if self._observation_dim is None:
            self._observation_dim = (
                self.config.embedding_dim +  # Instruction embedding
                self.config.num_objects * FEATURES_PER_OBJECT_TARGET  # All objects
            )
        return self._observation_dim

    def build(
        self,
        world_state: Dict[str, Any],
        intent_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Build observation for target selection.

        Args:
            world_state: From world.get_state()
            intent_embedding: 384-dim vector from TextEmbedder

        Returns:
            Fixed-length numpy array (564 floats)
        """
        obs_parts: List[np.ndarray] = []
        world_size = self.config.world_size

        # 1. Instruction embedding first (what we're trying to match)
        obs_parts.append(intent_embedding.astype(np.float32))

        # 2. All objects in consistent order (sorted by entity_id)
        objects = world_state["objects"]
        sorted_objects = sorted(objects, key=lambda o: o["entity_id"])

        for obj in sorted_objects:
            obj_features = self._encode_object(obj, world_size)
            obs_parts.append(obj_features)

        # Pad if fewer than expected objects
        num_missing = self.config.num_objects - len(sorted_objects)
        if num_missing > 0:
            zero_obj = np.zeros(FEATURES_PER_OBJECT_TARGET, dtype=np.float32)
            for _ in range(num_missing):
                obs_parts.append(zero_obj)

        # Concatenate all parts
        observation = np.concatenate(obs_parts)

        assert observation.shape[0] == self.observation_dim, \
            f"Observation dim mismatch: {observation.shape[0]} vs {self.observation_dim}"

        return observation

    def _encode_object(self, obj: Dict, world_size: float) -> np.ndarray:
        """
        Encode a single object for target selection.

        Features (18 total):
        - Absolute position normalized (2) - for context
        - Object type one-hot (3)
        - Color one-hot (6)
        - Shape one-hot (4)
        - Size one-hot (3)
        """
        ox, oy = obj["position"]["x"], obj["position"]["y"]

        features: List[float] = []

        # Position (normalized) - gives spatial context
        features.append(ox / world_size)
        features.append(oy / world_size)

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

        return np.array(features, dtype=np.float32)

    def _one_hot(self, value: str, categories: List[str]) -> List[float]:
        """Create a one-hot encoding for a categorical value."""
        result = [0.0] * len(categories)
        if value in categories:
            result[categories.index(value)] = 1.0
        return result

    def get_object_index_by_id(self, world_state: Dict[str, Any], entity_id: str) -> int:
        """
        Get the index (0-9) of an object in our consistent ordering.

        Useful for determining the "correct answer" during training.
        """
        objects = world_state["objects"]
        sorted_objects = sorted(objects, key=lambda o: o["entity_id"])

        for i, obj in enumerate(sorted_objects):
            if obj["entity_id"] == entity_id:
                return i
        return -1

    def get_object_id_by_index(self, world_state: Dict[str, Any], index: int) -> Optional[str]:
        """
        Get the entity_id of the object at a given index (0-9).

        Useful for translating agent's selection back to an object.
        """
        objects = world_state["objects"]
        sorted_objects = sorted(objects, key=lambda o: o["entity_id"])

        if 0 <= index < len(sorted_objects):
            return sorted_objects[index]["entity_id"]
        return None
