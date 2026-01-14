"""
PPO Trainer using stable-baselines3.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from .environment import NPCEnv
from src.runtime import RuntimeConfig
from src.reward import RewardConfig


@dataclass
class TrainerConfig:
    """Configuration for PPO training."""
    # Training params
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 2048          # Steps per environment per update
    batch_size: int = 64
    n_epochs: int = 10           # Epochs when optimizing surrogate loss
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # GAE lambda
    clip_range: float = 0.2      # PPO clip range
    ent_coef: float = 0.01       # Entropy coefficient (exploration)
    vf_coef: float = 0.5         # Value function coefficient
    max_grad_norm: float = 0.5   # Max gradient norm

    # Environment params
    n_envs: int = 4              # Parallel environments
    max_steps_per_episode: int = 500

    # Network architecture
    policy_type: str = "MlpPolicy"
    net_arch: Optional[Dict] = None  # Custom architecture

    # Saving/logging
    save_dir: str = "models"
    log_dir: str = "logs"
    save_freq: int = 10_000      # Save checkpoint every N steps
    eval_freq: int = 5_000       # Evaluate every N steps
    n_eval_episodes: int = 10

    # Runtime/reward config (passed to environment)
    runtime_config: Optional[RuntimeConfig] = None
    reward_config: Optional[RewardConfig] = None

    # Random seed
    seed: int = 42


class TrainingCallback(BaseCallback):
    """Custom callback for logging training progress."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode info if available
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]["episode"]
            self._episode_rewards.append(info["r"])
            self._episode_lengths.append(info["l"])

            if self.verbose > 0 and len(self._episode_rewards) % 10 == 0:
                avg_reward = sum(self._episode_rewards[-10:]) / 10
                avg_length = sum(self._episode_lengths[-10:]) / 10
                print(f"Episodes: {len(self._episode_rewards)}, "
                      f"Avg Reward (last 10): {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.1f}")

        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "episode_rewards": self._episode_rewards,
            "episode_lengths": self._episode_lengths,
        }


class Trainer:
    """
    PPO Trainer for the NPC environment.

    Usage:
        trainer = Trainer(config)
        trainer.train()
        trainer.save("my_model")

        # Later
        trainer.load("my_model")
        action = trainer.predict(observation)
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self._model: Optional[PPO] = None
        self._env = None
        self._eval_env = None
        self._callback: Optional[TrainingCallback] = None

        # Create directories
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def _make_env(self, seed: int, rank: int = 0) -> Callable:
        """Create a function that creates an environment."""
        def _init() -> NPCEnv:
            runtime_config = self.config.runtime_config or RuntimeConfig()
            if self.config.reward_config:
                runtime_config.reward_config = self.config.reward_config

            env = NPCEnv(
                runtime_config=runtime_config,
                max_steps_per_episode=self.config.max_steps_per_episode,
                seed=seed + rank,
            )
            return env
        return _init

    def setup(self) -> None:
        """Set up training environment and model."""
        set_random_seed(self.config.seed)

        # Create vectorized environment
        env_fns = [
            self._make_env(self.config.seed, i)
            for i in range(self.config.n_envs)
        ]

        if self.config.n_envs > 1:
            self._env = SubprocVecEnv(env_fns)
        else:
            self._env = DummyVecEnv(env_fns)

        # Create eval environment
        self._eval_env = DummyVecEnv([self._make_env(self.config.seed + 1000, 0)])

        # Define policy kwargs if custom architecture specified
        policy_kwargs = {}
        if self.config.net_arch:
            policy_kwargs["net_arch"] = self.config.net_arch

        # Create PPO model
        self._model = PPO(
            policy=self.config.policy_type,
            env=self._env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            verbose=1,
            tensorboard_log=self.config.log_dir,
            seed=self.config.seed,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
        )

        print(f"Model initialized with {self._count_parameters()} parameters")

    def _count_parameters(self) -> int:
        """Count trainable parameters in the model."""
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.policy.parameters() if p.requires_grad)

    def train(self, progress_bar: bool = True) -> Dict[str, Any]:
        """
        Train the model.

        Returns:
            Training statistics
        """
        if self._model is None:
            self.setup()

        # Set up callbacks
        self._callback = TrainingCallback(verbose=1)

        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq // self.config.n_envs,
            save_path=self.config.save_dir,
            name_prefix="ppo_npc",
        )

        eval_callback = EvalCallback(
            self._eval_env,
            best_model_save_path=os.path.join(self.config.save_dir, "best"),
            log_path=self.config.log_dir,
            eval_freq=self.config.eval_freq // self.config.n_envs,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
        )

        callbacks = [self._callback, checkpoint_callback, eval_callback]

        # Train
        print(f"Starting training for {self.config.total_timesteps} timesteps...")
        self._model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
        )

        return self._callback.get_stats()

    def save(self, name: str) -> str:
        """Save the model."""
        if self._model is None:
            raise RuntimeError("No model to save. Train first.")

        path = os.path.join(self.config.save_dir, name)
        self._model.save(path)
        print(f"Model saved to {path}")
        return path

    def load(self, path: str) -> None:
        """Load a saved model."""
        self._model = PPO.load(path)
        print(f"Model loaded from {path}")

    def predict(
        self,
        observation,
        deterministic: bool = True
    ) -> int:
        """
        Predict action for a single observation.

        Args:
            observation: The observation vector (575,)
            deterministic: If True, use deterministic action

        Returns:
            Action ID (0-4)
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Train or load first.")

        action, _ = self._model.predict(observation, deterministic=deterministic)
        return int(action)

    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the current model.

        Returns:
            Dict with mean_reward, std_reward, mean_length
        """
        if self._model is None:
            raise RuntimeError("No model to evaluate. Train or load first.")

        # Create fresh eval environment
        eval_env = NPCEnv(
            runtime_config=self.config.runtime_config,
            max_steps_per_episode=self.config.max_steps_per_episode,
            seed=self.config.seed + 9999,
        )

        rewards = []
        lengths = []

        for ep in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                if render:
                    eval_env.render()

            rewards.append(episode_reward)
            lengths.append(episode_length)

        eval_env.close()

        return {
            "mean_reward": sum(rewards) / len(rewards),
            "std_reward": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5,
            "mean_length": sum(lengths) / len(lengths),
            "n_episodes": n_episodes,
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._env:
            self._env.close()
        if self._eval_env:
            self._eval_env.close()
