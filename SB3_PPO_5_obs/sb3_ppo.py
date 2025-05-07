import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment import PointMassEnv

class EpisodeLogger(BaseCallback):
    """
    Stampa e salva ad ogni episodio:
      - numero episodio
      - reward totale
      - distanza (presa da info['distance'])
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards   = []
        self.episode_distances = []
        self.episode_count     = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_count += 1
                ep_reward = ep_info["r"]
                # prendo la distanza direttamente dall'info
                final_dist = info.get("distance", np.nan)

                self.episode_rewards.append(ep_reward)
                self.episode_distances.append(final_dist)

                print(f"Episode {self.episode_count} | Reward: {ep_reward:.2f} | Distance: {final_dist:.2f}")
        return True

if __name__ == "__main__":
    # 1) Env wrapped con Monitor (per episode info) e DummyVecEnv
    env = DummyVecEnv([lambda: Monitor(PointMassEnv())])

    # 2) Crea il modello PPO SB3
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        learning_rate=1e-4,
        ent_coef=0.02,
        clip_range=0.2,
        n_epochs=10,
        batch_size=64,
    )

    # 3) Callback di logging
    callback = EpisodeLogger()

    # 4) Training (500 episodi × max 500 step)
    total_timesteps = 500 * 500
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 5) Plot dei risultati
    episodes = np.arange(1, len(callback.episode_rewards) + 1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(episodes, callback.episode_rewards, marker='o')
    axs[0].set(xlabel="Episode", ylabel="Total Reward", title="Total Reward per Episode")

    axs[1].plot(episodes, callback.episode_distances, marker='o', color='orange')
    axs[1].set(xlabel="Episode", ylabel="Final Distance", title="Final Distance from Target per Episode")

    plt.tight_layout()
    plt.show()

    # 6) Salva il modello
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"ppo_sb3_{ts}.zip")
    model.save(model_path)

    print("\nTraining completed!")