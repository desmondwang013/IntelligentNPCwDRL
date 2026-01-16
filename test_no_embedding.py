"""
Test navigation WITHOUT the 384-dim embedding.
This isolates whether the embedding complexity is blocking learning.

Observation without embedding:
- NPC position (2)
- User position (2)
- NPC→User offset (2)
- Objects (10 × 22 = 220)
- Intent age (1)
- Focus hint (10)
Total: 237 dimensions (vs 621 with embedding)
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.training import NPCEnv


class NoEmbeddingEnv(gym.Wrapper):
    """
    Wraps NPCEnv but removes the 384-dim embedding from observations.
    Everything else stays the same.
    """

    EMBEDDING_DIM = 384

    def __init__(self, env):
        super().__init__(env)

        # Calculate new observation size (original minus embedding)
        original_dim = env.observation_space.shape[0]
        self.new_dim = original_dim - self.EMBEDDING_DIM

        # New observation space without embedding
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.new_dim,),
            dtype=np.float32
        )

        # Find where embedding starts and ends in the observation
        # Structure: npc(2) + user(2) + offset(2) + objects(220) + embedding(384) + age(1) + focus(10)
        self.embedding_start = 2 + 2 + 2 + 220  # = 226
        self.embedding_end = self.embedding_start + self.EMBEDDING_DIM  # = 610

    def _remove_embedding(self, obs):
        """Remove the embedding portion from observation."""
        before = obs[:self.embedding_start]
        after = obs[self.embedding_end:]
        return np.concatenate([before, after])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._remove_embedding(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._remove_embedding(obs), reward, terminated, truncated, info


def make_env():
    """Create a no-embedding environment."""
    def _init():
        env = NPCEnv()
        env = NoEmbeddingEnv(env)
        return env
    return _init


def main():
    print("=" * 60)
    print("Testing Navigation WITHOUT Embedding")
    print("=" * 60)
    print(f"Original observation: 621 dimensions")
    print(f"Without embedding:    237 dimensions")
    print("=" * 60)

    # Create vectorized environments
    n_envs = 4
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env()])

    # Create model with same hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu",
    )

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/test_no_embedding/",
        log_path="./logs/test_no_embedding/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # Train
    print("\nTraining for 100k steps...")
    model.learn(total_timesteps=100000, callback=eval_callback, progress_bar=True)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (20 episodes)")
    print("=" * 60)

    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Also check episode lengths manually
    episode_lengths = []
    obs = eval_env.reset()
    for _ in range(20):
        done = False
        length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            length += 1
            if done:
                episode_lengths.append(length)
                obs = eval_env.reset()

    mean_length = np.mean(episode_lengths)
    print(f"Mean episode length: {mean_length:.1f}")

    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    if mean_length < 400:
        print("PROGRESS! Agent is completing some tasks.")
    else:
        print("Still timing out. Problem is not just the embedding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
