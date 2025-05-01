import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env_sb3_ppo_8_obs import PointMassEnv

class EpisodeLogger:
    def __init__(self):
        self.rewards   = []
        self.distances = []

    def __call__(self, locals_, globals_):
        infos = locals_["infos"]
        for info in infos:
            ep = info.get("episode")
            if ep:
                r = ep["r"]
                d = info.get("distance", np.nan)
                self.rewards.append(r)
                self.distances.append(d)
                n = len(self.rewards)
                print(f"Episode {n} | Reward: {r:.2f} | Distance: {d:.2f}")
        return True


# 1) crea env SB3-compatibile
env = DummyVecEnv([lambda: Monitor(PointMassEnv())])

# 2) modello PPO con policy MLP standard
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    ent_coef=0.02,
    clip_range=0.2,
    n_epochs=10,
    batch_size=64,
    verbose=0,
)

# 3) callback a funzione
callback = EpisodeLogger()

# 4) training
total_timesteps = 500 * 500
model.learn(total_timesteps=total_timesteps, callback=callback)

# 5) plot
episodes = np.arange(1, len(callback.rewards)+1)
fig, axs = plt.subplots(2,1,figsize=(8,6))
axs[0].plot(episodes, callback.rewards,  marker='o')
axs[0].set(xlabel="Episode", ylabel="Total Reward")
axs[1].plot(episodes, callback.distances, marker='o', color='orange')
axs[1].set(xlabel="Episode", ylabel="Final Distance")
plt.tight_layout()
plt.show()

# 6) salva modello
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs("models", exist_ok=True)
model.save(f"models/ppo_sb3_8_obs_{ts}.zip")
print("\nTraining completed!")