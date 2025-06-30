import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from PursuitChase import ChaseEnv

class EpisodeLogger:
    def __init__(self):
        self.rewards = []
        self.statuses = []

    def __call__(self, locals_, globals_):
        succ_count = 0
        fail_count = 0
        infos = locals_["infos"]
        for i, done in enumerate(infos):
            if done and infos[i].get("episode") is not None:
                ep      = infos[i]["episode"]
                r       = ep["r"]
                success = infos[i].get("is_success", False)
                fail    = infos[i].get("is_failure", False)
                status  = "Success" if success else "Fail" if fail else "Unknown"
                idx = len(self.rewards) + 1
                self.rewards.append(r)
                self.statuses.append(status)

                print(f"Episode {idx} | Reward: {r:.2f} | Status: {status}")
        return True
    
def lr_schedule(progress): return 3e-4 * (1 - progress) # learning rate schedule
    
def make_env():
    def _init():
        return ChaseEnv()
    return _init

env = DummyVecEnv([make_env() for _ in range(10)])
env = VecMonitor(env)

model = PPO(
    "MlpPolicy", env,
    policy_kwargs=dict(net_arch=[64, 64]),
    learning_rate=lr_schedule,
    ent_coef=1e-3,
    clip_range=0.2,
    n_epochs=20,
    batch_size=256,
    verbose=0,
    device="cpu"
)

callback = EpisodeLogger()
total_timesteps = 500 * 2000
model.learn(total_timesteps=total_timesteps, callback=callback)

successes = callback.statuses.count("Success")
failures  = callback.statuses.count("Fail")
total_eps = len(callback.statuses)

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs('models', exist_ok=True)
model.save(f'models/ppo_UCAV_{ts}.zip')

print(f"Out of {total_eps} episodes: {successes} successes, {failures} failures")
print("\nTraining completed!")