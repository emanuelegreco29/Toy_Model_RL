import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env_fight import PointMassEnv

class EpisodeLogger:
    def __init__(self):
        self.rewards = []
        self.distances = []
        self.followed = []
        self.final_hps = []

    def __call__(self, locals_, globals_):
        for info in locals_["infos"]:
            ep = info.get("episode")
            if ep:
                r = ep["r"]
                d = info.get("distance", np.nan)
                f = info.get("followed", np.nan)
                hp = info.get("final_hp", np.nan)
                n = len(self.rewards) + 1
                self.rewards.append(r)
                self.distances.append(d)
                self.followed.append(f)
                self.final_hps.append(hp)
                print(f"Episode {n} | Reward: {r:.2f} | Followed: {f} | Distance: {d:.2f} | Final HP: {hp}")
        return True

# Definizione dell'architettura della policy
args = dict(
    net_arch=[512, 512, 512],
    log_std_init=-1.5
)


env = DummyVecEnv([lambda: Monitor(PointMassEnv(K_history=1))])
model = PPO(
    "MlpPolicy", env,
    policy_kwargs=args,
    learning_rate=1e-4,
    ent_coef=0.01,
    clip_range=0.2,
    n_epochs=20,
    batch_size=512,
    verbose=0,
    device="cpu"
)

callback = EpisodeLogger()
total_timesteps = PointMassEnv.max_steps * 3000
model.learn(total_timesteps=total_timesteps, callback=callback)

eps = np.arange(1, len(callback.rewards) + 1)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
ax1.plot(eps, callback.rewards, marker='o')
ax1.set(xlabel='Episode', ylabel='Reward')
ax2.plot(eps, callback.followed, marker='o')
ax2.set(xlabel='Episode', ylabel='Followed steps')
ax3.plot(eps, callback.final_hps, marker='o')
ax3.set(xlabel='Episode', ylabel='Final HP')
plt.tight_layout()
plt.show()

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs('models', exist_ok=True)
model.save(f'models/ppo_fight_{ts}.zip')
print("\nTraining completed!")
