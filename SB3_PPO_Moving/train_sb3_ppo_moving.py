import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env_sb3_ppo_moving import PointMassEnv

class EpisodeLogger:
    def __init__(self):
        self.rewards = []
        self.distances = []
        
    def __call__(self, locals_, globals_):
        for info in locals_["infos"]:
            ep = info.get("episode")
            if ep:
                r = ep["r"]
                d = info.get("distance", np.nan)
                n = len(self.rewards) + 1
                self.rewards.append(r)
                self.distances.append(d)
                print(f"Episode {n} | Reward: {r:.2f} | Distance: {d:.2f}")
        return True

env = DummyVecEnv([lambda: Monitor(PointMassEnv())])
model = PPO(
    "MlpPolicy", env,
    learning_rate=1e-4,
    ent_coef=0.02,
    clip_range=0.2,
    n_epochs=10,
    batch_size=64,
    verbose=0
)
callback = EpisodeLogger()
total_timesteps = 500 * 1000
model.learn(total_timesteps=total_timesteps, callback=callback)

eps = np.arange(1, len(callback.rewards) + 1)
fig, axs = plt.subplots(2,1, figsize=(8,6))
axs[0].plot(eps, callback.rewards, marker='o')
axs[0].set(xlabel='Episode', ylabel='Reward')
axs[1].plot(eps, callback.distances, marker='o', color='orange')
axs[1].set(xlabel='Episode', ylabel='Distance')
plt.tight_layout()
plt.show()

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs('models', exist_ok=True)
model.save(f'models/ppo_sb3_moving_target_{ts}.zip')
print("\nTraining completed!")