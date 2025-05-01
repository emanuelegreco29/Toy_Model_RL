import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PointMassEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(PointMassEnv, self).__init__()
        self.dt = 0.1
        self.max_steps = 500
        self.current_step = 0

        # limiti per Δθ, Δz
        self.delta_theta = 0.5
        self.delta_z     = 0.25

        # Action space continuo: [Δθ, Δz]
        low  = np.array([-self.delta_theta, -self.delta_z], dtype=np.float32)
        high = np.array([ self.delta_theta,  self.delta_z], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # **OBSERVATION SPACE 8-D**: [x,y,z,v,θ, x_t,y_t,z_t]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # stato iniziale e target di default
        self.state  = np.array([0.0,0.0,0.0, 1.0,0.0], dtype=np.float32)
        self.target = np.array([10.0,10.0, 5.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.state = np.array([0.0,0.0,0.0,1.0,0.0], dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        # restituisci [x,y,z,v,θ, x_t,y_t,z_t]
        return np.concatenate([ self.state, self.target ]).astype(np.float32)

    def compute_reward(self, prev_state, current_state, target):
        curr_pos    = current_state[:3]
        prev_pos    = prev_state[:3]
        curr_dist   = np.linalg.norm(curr_pos - target)
        prev_dist   = np.linalg.norm(prev_pos - target)
        improvement = prev_dist - curr_dist

        if improvement > 0:
            reward = 1.0 * improvement
        else:
            reward = -0.5 * abs(improvement)
        reward -= 0.001
        
        if(curr_dist < 0.5):
            reward += 50.0
        
        return reward

    def step(self, action):
        x, y, z, v, theta = self.state
        dtheta, dz = action
        # normalizza theta in [-π,π]
        theta = (theta + dtheta + np.pi) % (2*np.pi) - np.pi
        # aggiorna z
        z = z + dz

        x = x + v * np.cos(theta) * self.dt
        y = y + v * np.sin(theta) * self.dt

        prev_state = self.state.copy()
        self.state = np.array([x,y,z,v,theta], dtype=np.float32)
        self.current_step += 1

        reward = self.compute_reward(prev_state, self.state, self.target)

        terminated = np.linalg.norm(self.state[:3] - self.target) < 0.5
        truncated  = self.current_step >= self.max_steps

        # infilo la distanza in info
        info = {"distance": float(np.linalg.norm(self.state[:3] - self.target))}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        print("Pos:", self.state[:3], "Target:", self.target)