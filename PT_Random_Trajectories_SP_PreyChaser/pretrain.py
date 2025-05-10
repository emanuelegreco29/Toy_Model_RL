import gymnasium as gym
import numpy as np
import os
import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import env_PT_rand_traj

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if done:
                    self.episode_count += 1
                    ep_info    = infos[idx].get("episode", {})
                    ep_reward  = ep_info.get("r", float("nan"))
                    final_dist = infos[idx].get("distance", float("nan"))
                    print(f"Episode {self.episode_count} | Reward: {ep_reward:.2f} | Final distance: {final_dist:.2f}")
        return True

class PretrainWrapper(gym.Wrapper):
    """
    Wrapper for pretraining:
    - fixes target dv to zero
    - augments observation with next K target waypoints
    """
    def __init__(self, env, K=20):
        super().__init__(env)
        self.K = K
        # augment observation: original 10 + 3*K future coords
        low = np.concatenate([env.observation_space.low, np.full((3*K,), -np.inf)], dtype=np.float32)
        high= np.concatenate([env.observation_space.high, np.full((3*K,),  np.inf)], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        # agent-only discrete actions
        self.action_space = gym.spaces.Discrete(env.n_theta * env.n_z * env.n_v)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        # decode discrete agent action
        idx = int(action)
        idx_th = idx // (self.env.n_z * self.env.n_v)
        idx_z  = (idx // self.env.n_v) % self.env.n_z
        idx_v  = idx % self.env.n_v
        dtheta = self.env.theta_bins[idx_th]
        dz     = self.env.z_bins[idx_z]
        dv     = self.env.v_bins[idx_v]
        action_agent = np.array([dtheta, dz, dv], dtype=np.float32)
        action_target = np.array([0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env._step_continuous(action_agent, action_target)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs):
        idx = self.env.current_step
        traj = self.env.target_trajectory
        future = traj[idx:idx+self.K]
        if future.shape[0] < self.K:
            pad = np.repeat(traj[-1][None], self.K - future.shape[0], axis=0)
            future = np.vstack([future, pad])
        return np.concatenate([obs, future.flatten()]).astype(np.float32)


def make_pretrain_env():
    env = PretrainWrapper(env_PT_rand_traj.DiscretePointMassEnv(), K=20)
    return Monitor(env)

vec_env = make_vec_env(make_pretrain_env, n_envs=4)

model = DQN(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=0
)

model.learn(total_timesteps=500000, callback=EpisodeLoggerCallback())
ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs('models', exist_ok=True)
fname = f'models/dqn_pretrain_{ts}.zip'
model.save(fname)