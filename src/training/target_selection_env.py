"""
Gym environment for training the target selection policy.

This is a single-step environment: the agent sees an instruction
embedding + similarity scores to each object + object features,
then selects one object.

Key insight: We compute cosine similarity between instruction embedding
and each object's description embedding. This gives the agent a clear
signal in the same semantic space.
"""
import random
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.world import World, WorldConfig
from src.intent import TextEmbedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class TargetSelectionEnv(gym.Env):
    """
    Environment for learning target selection from instructions.

    The key improvement: we embed object descriptions and compute
    similarity scores, giving the agent a learnable signal.

    Observation structure:
    - Similarity scores (10): cosine similarity between instruction and each object
    - Object features (10 × 18 = 180): position + type/color/shape/size one-hots

    Total: 190 floats (much smaller than before!)

    Episode structure (single step):
    1. Reset: generate random world + instruction
    2. Agent sees: similarity scores + object features
    3. Agent outputs: object index (0-9)
    4. Reward: 1.0 if correct, 0.0 if wrong
    5. Episode ends immediately

    Action space: Discrete(10) - which object to select
    """

    metadata = {"render_modes": ["human"]}

    # One-hot encoding sizes
    NUM_OBJECT_TYPES = 3  # fixed, light, heavy
    NUM_COLORS = 6
    NUM_SHAPES = 4
    NUM_SIZES = 3
    FEATURES_PER_OBJECT = 2 + NUM_OBJECT_TYPES + NUM_COLORS + NUM_SHAPES + NUM_SIZES  # 18

    def __init__(
        self,
        world_config: Optional[WorldConfig] = None,
        instruction_templates: Optional[List[str]] = None,
        seed: Optional[int] = None,
        num_objects: int = 10,
    ):
        super().__init__()

        self._world_config = world_config or WorldConfig()
        self._seed = seed
        self._rng = random.Random(seed)
        self._num_objects = num_objects

        # Instruction templates for generating random tasks
        self._instruction_templates = instruction_templates or [
            "Go to the {color} {shape}",
            "Move to the {color} {shape}",
            "Walk to the {color} {shape}",
            "Find the {color} {shape}",
            "Head to the {color} {shape}",
        ]

        # Components
        self._world: Optional[World] = None
        self._embedder = TextEmbedder()

        # Cache for object description embeddings
        self._object_embeddings_cache: Dict[str, np.ndarray] = {}

        # Current episode state
        self._current_instruction: Optional[str] = None
        self._current_embedding: Optional[np.ndarray] = None
        self._correct_object_index: Optional[int] = None
        self._correct_object_id: Optional[str] = None
        self._similarity_scores: Optional[np.ndarray] = None

        # Define spaces
        self.action_space = spaces.Discrete(num_objects)
        obs_dim = num_objects + (num_objects * self.FEATURES_PER_OBJECT)  # 10 + 180 = 190
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_object_embedding(self, obj: Dict) -> np.ndarray:
        """Get embedding for an object's description, with caching."""
        description = f"{obj['color']} {obj['shape']}"
        if description not in self._object_embeddings_cache:
            self._object_embeddings_cache[description] = self._embedder.embed(description)
        return self._object_embeddings_cache[description]

    def _encode_object(self, obj: Dict, world_size: float = 64.0) -> np.ndarray:
        """Encode object features (position + one-hots)."""
        features: List[float] = []

        # Position (normalized)
        features.append(obj["position"]["x"] / world_size)
        features.append(obj["position"]["y"] / world_size)

        # Type one-hot
        types = ["fixed", "light", "heavy"]
        features.extend([1.0 if obj["object_type"] == t else 0.0 for t in types])

        # Color one-hot
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        features.extend([1.0 if obj["color"] == c else 0.0 for c in colors])

        # Shape one-hot
        shapes = ["circle", "triangle", "square", "diamond"]
        features.extend([1.0 if obj["shape"] == s else 0.0 for s in shapes])

        # Size one-hot
        sizes = ["small", "medium", "large"]
        features.extend([1.0 if obj["size"] == s else 0.0 for s in sizes])

        return np.array(features, dtype=np.float32)

    def _build_observation(self, world_state: Dict) -> np.ndarray:
        """Build observation with similarity scores + object features."""
        objects = sorted(world_state["objects"], key=lambda o: o["entity_id"])

        obs_parts: List[np.ndarray] = []

        # Part 1: Similarity scores (10 floats)
        similarities = []
        for obj in objects:
            obj_emb = self._get_object_embedding(obj)
            sim = cosine_similarity(self._current_embedding, obj_emb)
            similarities.append(sim)
        self._similarity_scores = np.array(similarities, dtype=np.float32)
        obs_parts.append(self._similarity_scores)

        # Part 2: Object features (10 × 18 = 180 floats)
        for obj in objects:
            obj_features = self._encode_object(obj)
            obs_parts.append(obj_features)

        # Pad if fewer objects
        num_missing = self._num_objects - len(objects)
        if num_missing > 0:
            # Zero similarity for missing objects
            obs_parts[0] = np.pad(self._similarity_scores, (0, num_missing))
            # Zero features for missing objects
            for _ in range(num_missing):
                obs_parts.append(np.zeros(self.FEATURES_PER_OBJECT, dtype=np.float32))

        return np.concatenate(obs_parts)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset for a new episode."""
        if seed is not None:
            self._rng = random.Random(seed)
            self._seed = seed

        # Create fresh world
        world_seed = self._rng.randint(0, 1000000)
        self._world_config.random_seed = world_seed
        self._world = World(config=self._world_config)

        # Generate random instruction
        self._generate_instruction()

        # Build observation
        world_state = self._world.get_state()
        obs = self._build_observation(world_state)

        info = {
            "instruction": self._current_instruction,
            "correct_object_id": self._correct_object_id,
            "correct_object_index": self._correct_object_index,
            "similarity_scores": self._similarity_scores.tolist(),
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step (the only step in this episode)."""
        if self._world is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Check if selection is correct
        is_correct = (action == self._correct_object_index)
        reward = 1.0 if is_correct else 0.0

        # Get what the agent selected
        world_state = self._world.get_state()
        objects = sorted(world_state["objects"], key=lambda o: o["entity_id"])

        selected_obj = objects[action] if action < len(objects) else None

        # Build info
        info = {
            "instruction": self._current_instruction,
            "correct_object_id": self._correct_object_id,
            "correct_object_index": self._correct_object_index,
            "selected_object_index": action,
            "is_correct": is_correct,
            "similarity_scores": self._similarity_scores.tolist(),
        }

        if selected_obj:
            info["selected_object_id"] = selected_obj["entity_id"]
            info["selected_description"] = f"{selected_obj['color']} {selected_obj['shape']}"
            info["selected_similarity"] = self._similarity_scores[action]

        # Return same observation (episode ends anyway)
        obs = self._build_observation(world_state)

        return obs, reward, True, False, info

    def _generate_instruction(self) -> None:
        """Generate a random instruction targeting one of the objects."""
        world_state = self._world.get_state()
        objects = world_state["objects"]

        # Pick a random object as the target
        target = self._rng.choice(objects)
        self._correct_object_id = target["entity_id"]

        # Find its index in sorted order
        sorted_objects = sorted(objects, key=lambda o: o["entity_id"])
        for i, obj in enumerate(sorted_objects):
            if obj["entity_id"] == self._correct_object_id:
                self._correct_object_index = i
                break

        # Generate instruction text
        template = self._rng.choice(self._instruction_templates)
        self._current_instruction = template.format(
            color=target["color"],
            shape=target["shape"],
        )

        # Embed the instruction
        self._current_embedding = self._embedder.embed(self._current_instruction)

    def render(self) -> None:
        """Render the current state."""
        if self._world is None:
            print("Environment not initialized.")
            return

        print(f"Instruction: {self._current_instruction}")
        print(f"Correct object: index={self._correct_object_index}, id={self._correct_object_id}")

        world_state = self._world.get_state()
        print("\nObjects with similarity scores:")
        sorted_objects = sorted(world_state["objects"], key=lambda o: o["entity_id"])
        for i, obj in enumerate(sorted_objects):
            sim = self._similarity_scores[i] if self._similarity_scores is not None else 0
            marker = " <-- CORRECT" if i == self._correct_object_index else ""
            print(f"  [{i}] {obj['color']} {obj['shape']} (sim={sim:.3f}){marker}")

    def close(self) -> None:
        """Clean up resources."""
        self._world = None

    @property
    def embedder(self) -> TextEmbedder:
        """Access the text embedder."""
        return self._embedder
