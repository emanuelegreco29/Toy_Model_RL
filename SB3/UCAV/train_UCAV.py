import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from PursuitChase import ChaseEnv

class EpisodeLogger:
    def __init__(self):
        self.rewards = []
        self.statuses = []

    def __call__(self, locals_, globals_):
        for info in locals_["infos"]:
            ep = info.get("episode")
            if ep is not None:
                r   = ep["r"]
                # prendo i flag che ChaseEnv popola in info
                success = info.get("is_success", False)
                fail    = info.get("is_failure", False)
                status  = "Success" if success else "Fail" if fail else "Unknown"

                idx = len(self.rewards) + 1
                self.rewards.append(r)
                self.statuses.append(status)

                print(f"Episode {idx} | Reward: {r:.2f} | Status: {status}")
        return True
    
def make_env():
    def _init():
        return ChaseEnv()
    return _init

env = DummyVecEnv([make_env() for _ in range(4)])
env = VecMonitor(env)

model = PPO(
    "MlpPolicy", env,
    policy_kwargs=dict(net_arch=[128, 128]),
    learning_rate=3e-5,
    ent_coef=0.0,
    clip_range=0.1,
    n_epochs=20,
    batch_size=256,
    verbose=0,
    device="cpu"
)

callback = EpisodeLogger()
total_timesteps = 500 * 10
model.learn(total_timesteps=total_timesteps, callback=callback)

eps = np.arange(1, len(callback.rewards) + 1)
fig, axs = plt.subplots(2,1, figsize=(8,6))
axs[0].plot(eps, callback.rewards, marker='o')
axs[0].set(xlabel='Episode', ylabel='Reward')
axs[1].plot(eps, callback.followed, marker='o', color='orange')
axs[1].set(xlabel='Episode', ylabel='Followed target for X steps')
plt.tight_layout()
plt.show()

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs('models', exist_ok=True)
model.save(f'models/ppo_UCAV_{ts}.zip')
print("\nTraining completed!")