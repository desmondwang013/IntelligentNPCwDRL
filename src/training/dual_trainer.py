"""
Trainer for the two-policy architecture.

Trains:
1. Target Selection Policy: instruction → which object (Discrete 10)
2. Movement Policy: state → movement action (Discrete 5)

These are trained separately but used together at inference time.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np

from .target_selection_env import TargetSelectionEnv
from .movement_env import MovementEnv


@dataclass
class DualTrainerConfig:
    """Configuration for dual policy training."""
    # Target selection training
    target_timesteps: int = 50000
    target_n_envs: int = 4
    target_learning_rate: float = 3e-4

    # Movement training
    movement_timesteps: int = 100000
    movement_n_envs: int = 4
    movement_learning_rate: float = 3e-4

    # Shared settings
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5

    # Directories
    save_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    n_eval_episodes: int = 10


class DualPolicyTrainer:
    """
    Trains both target selection and movement policies.

    Training strategy:
    1. Train target selection policy (simpler task)
    2. Train movement policy (using random target indices)
    3. Save both models

    At inference:
    1. Target policy selects object from instruction
    2. Movement policy navigates to that object
    """

    def __init__(self, config: Optional[DualTrainerConfig] = None):
        self.config = config or DualTrainerConfig()

        self._target_model: Optional[PPO] = None
        self._movement_model: Optional[PPO] = None

        self._target_envs = None
        self._movement_envs = None

        # Create directories
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def setup_target_training(self) -> None:
        """Set up environments and model for target selection training."""
        print("Setting up target selection training...")

        def make_env():
            return TargetSelectionEnv()

        # Create vectorized environments
        if self.config.target_n_envs > 1:
            self._target_envs = SubprocVecEnv([make_env for _ in range(self.config.target_n_envs)])
        else:
            self._target_envs = DummyVecEnv([make_env])

        # Create PPO model for target selection
        self._target_model = PPO(
            "MlpPolicy",
            self._target_envs,
            learning_rate=self.config.target_learning_rate,
            n_steps=128,  # Shorter since episodes are 1 step
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ent_coef=self.config.ent_coef,
            max_grad_norm=self.config.max_grad_norm,
            vf_coef=self.config.vf_coef,
            tensorboard_log=f"{self.config.log_dir}/target_selection",
            verbose=1,
        )

        print(f"Target selection model initialized")
        print(f"  Observation space: {self._target_envs.observation_space}")
        print(f"  Action space: {self._target_envs.action_space}")

    def setup_movement_training(self) -> None:
        """Set up environments and model for movement training."""
        print("Setting up movement training...")

        def make_env():
            return MovementEnv()

        # Create vectorized environments
        if self.config.movement_n_envs > 1:
            self._movement_envs = SubprocVecEnv([make_env for _ in range(self.config.movement_n_envs)])
        else:
            self._movement_envs = DummyVecEnv([make_env])

        # Create PPO model for movement
        self._movement_model = PPO(
            "MlpPolicy",
            self._movement_envs,
            learning_rate=self.config.movement_learning_rate,
            n_steps=2048,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ent_coef=self.config.ent_coef,
            max_grad_norm=self.config.max_grad_norm,
            vf_coef=self.config.vf_coef,
            tensorboard_log=f"{self.config.log_dir}/movement",
            verbose=1,
        )

        print(f"Movement model initialized")
        print(f"  Observation space: {self._movement_envs.observation_space}")
        print(f"  Action space: {self._movement_envs.action_space}")

    def train_target_selection(self, progress_bar: bool = True) -> None:
        """Train the target selection policy."""
        if self._target_model is None:
            self.setup_target_training()

        print(f"\n{'='*50}")
        print("Training target selection policy...")
        print(f"  Timesteps: {self.config.target_timesteps}")
        print(f"{'='*50}\n")

        # Callbacks
        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=max(self.config.checkpoint_freq // self.config.target_n_envs, 1),
            save_path=f"{self.config.save_dir}/target_checkpoints",
            name_prefix="target_policy",
        )
        callbacks.append(checkpoint_callback)

        # Train
        self._target_model.learn(
            total_timesteps=self.config.target_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
        )

        print("Target selection training complete!")

    def train_movement(self, progress_bar: bool = True) -> None:
        """Train the movement policy."""
        if self._movement_model is None:
            self.setup_movement_training()

        print(f"\n{'='*50}")
        print("Training movement policy...")
        print(f"  Timesteps: {self.config.movement_timesteps}")
        print(f"{'='*50}\n")

        # Callbacks
        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=max(self.config.checkpoint_freq // self.config.movement_n_envs, 1),
            save_path=f"{self.config.save_dir}/movement_checkpoints",
            name_prefix="movement_policy",
        )
        callbacks.append(checkpoint_callback)

        # Train
        self._movement_model.learn(
            total_timesteps=self.config.movement_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
        )

        print("Movement training complete!")

    def train_both(self, progress_bar: bool = True) -> None:
        """Train both policies in sequence."""
        print("\n" + "="*60)
        print("DUAL POLICY TRAINING")
        print("="*60)

        # Phase 1: Target selection
        self.train_target_selection(progress_bar=progress_bar)

        # Phase 2: Movement
        self.train_movement(progress_bar=progress_bar)

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)

    def save(self, name: str = "dual_policy") -> None:
        """Save both models."""
        save_dir = Path(self.config.save_dir)

        if self._target_model:
            target_path = save_dir / f"{name}_target"
            self._target_model.save(str(target_path))
            print(f"Target model saved to: {target_path}")

        if self._movement_model:
            movement_path = save_dir / f"{name}_movement"
            self._movement_model.save(str(movement_path))
            print(f"Movement model saved to: {movement_path}")

    def load(self, name: str = "dual_policy") -> None:
        """Load both models."""
        save_dir = Path(self.config.save_dir)

        target_path = save_dir / f"{name}_target.zip"
        if target_path.exists():
            # Need a dummy env for loading
            dummy_env = DummyVecEnv([lambda: TargetSelectionEnv()])
            self._target_model = PPO.load(str(target_path), env=dummy_env)
            print(f"Target model loaded from: {target_path}")
        else:
            print(f"Target model not found at: {target_path}")

        movement_path = save_dir / f"{name}_movement.zip"
        if movement_path.exists():
            dummy_env = DummyVecEnv([lambda: MovementEnv()])
            self._movement_model = PPO.load(str(movement_path), env=dummy_env)
            print(f"Movement model loaded from: {movement_path}")
        else:
            print(f"Movement model not found at: {movement_path}")

    def predict_target(self, observation: np.ndarray) -> int:
        """Use target policy to select an object."""
        if self._target_model is None:
            raise RuntimeError("Target model not loaded/trained")

        action, _ = self._target_model.predict(observation, deterministic=True)
        return int(action)

    def predict_movement(self, observation: np.ndarray) -> int:
        """Use movement policy to select a movement action."""
        if self._movement_model is None:
            raise RuntimeError("Movement model not loaded/trained")

        action, _ = self._movement_model.predict(observation, deterministic=True)
        return int(action)

    def evaluate_target_selection(self, n_episodes: int = 100) -> Dict:
        """Evaluate target selection accuracy."""
        if self._target_model is None:
            raise RuntimeError("Target model not loaded/trained")

        env = TargetSelectionEnv()
        correct = 0

        for _ in range(n_episodes):
            obs, info = env.reset()
            action, _ = self._target_model.predict(obs, deterministic=True)

            _, _, _, _, step_info = env.step(int(action))
            if step_info["is_correct"]:
                correct += 1

        accuracy = correct / n_episodes
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": n_episodes,
        }

    def evaluate_movement(self, n_episodes: int = 20) -> Dict:
        """Evaluate movement policy success rate."""
        if self._movement_model is None:
            raise RuntimeError("Movement model not loaded/trained")

        env = MovementEnv()
        successes = 0
        total_steps = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            steps = 0

            while not done:
                action, _ = self._movement_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(int(action))
                done = terminated or truncated
                steps += 1

            if step_info.get("termination_reason") == "INTENT_COMPLETED":
                successes += 1
            total_steps.append(steps)

        return {
            "success_rate": successes / n_episodes,
            "successes": successes,
            "total": n_episodes,
            "avg_steps": np.mean(total_steps),
        }

    def evaluate_combined(self, n_episodes: int = 20) -> Dict:
        """
        Evaluate both policies working together.

        Flow:
        1. Generate instruction
        2. Target policy selects object
        3. Movement policy navigates
        4. Check if correct object was reached
        """
        if self._target_model is None or self._movement_model is None:
            raise RuntimeError("Both models must be loaded/trained")

        # We need both envs
        target_env = TargetSelectionEnv()
        movement_env = MovementEnv()

        correct_selection = 0
        successful_navigation = 0
        full_success = 0

        for _ in range(n_episodes):
            # Reset target env to get instruction + world
            target_obs, target_info = target_env.reset()

            # Target policy selects object
            selected_idx, _ = self._target_model.predict(target_obs, deterministic=True)
            selected_idx = int(selected_idx)

            is_correct_selection = (selected_idx == target_info["correct_object_index"])
            if is_correct_selection:
                correct_selection += 1

            # Movement env with selected target
            # Note: This creates a new world, so we're testing generalization
            # For true combined eval, we'd need same world state
            movement_obs, _ = movement_env.reset(options={"target_index": selected_idx})

            done = False
            while not done:
                action, _ = self._movement_model.predict(movement_obs, deterministic=True)
                movement_obs, _, terminated, truncated, step_info = movement_env.step(int(action))
                done = terminated or truncated

            if step_info.get("termination_reason") == "INTENT_COMPLETED":
                successful_navigation += 1
                if is_correct_selection:
                    full_success += 1

        return {
            "target_accuracy": correct_selection / n_episodes,
            "navigation_success": successful_navigation / n_episodes,
            "full_success": full_success / n_episodes,
            "episodes": n_episodes,
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._target_envs:
            self._target_envs.close()
        if self._movement_envs:
            self._movement_envs.close()
