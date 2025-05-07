import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from env_sb3_dqn_moving import DiscretePointMassEnv

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

env = Monitor(DiscretePointMassEnv(n_theta=3, n_z=3))
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    target_update_interval=500,
    train_freq=4,
    verbose=0
)

callback = EpisodeLogger()
total_timesteps = 500 * 1000
model.learn(total_timesteps=total_timesteps, callback=callback)

# plot dell’apprendimento
eps = np.arange(1, len(callback.rewards) + 1)
fig, axs = plt.subplots(2,1, figsize=(8,6))
axs[0].plot(eps, callback.rewards, marker='o')
axs[0].set(xlabel='Episode', ylabel='Reward')
axs[1].plot(eps, callback.distances, marker='o')
axs[1].set(xlabel='Episode', ylabel='Distance')
plt.tight_layout()
plt.show()

# salva modello
ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs('models', exist_ok=True)
fname = f'models/dqn_sb3_moving_target_{ts}.zip'
model.save(fname)
print(f"\nTraining completed! Model salvato in {fname}")