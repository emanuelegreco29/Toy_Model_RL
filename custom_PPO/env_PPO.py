import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PointMassEnv(gym.Env):
    dt = 0.1
    max_steps = 500

    def __init__(self):
        super().__init__()
        self.current_step = 0

        self.delta_theta = 0.5
        self.delta_z     = 0.25
        
        # Action: [Δθ, Δz] (2 azioni continue)
        low_act  = np.array([-self.delta_theta, -self.delta_z], dtype=np.float32)
        high_act = np.array([ self.delta_theta,  self.delta_z], dtype=np.float32)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        # Observation: 5 (agent state) + 3 (target position)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0

        # Stato iniziale agente: [x, y, z, v, θ]
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        # Stato target: [x_t, y_t, z_t, v_t, θ_t]
        self.target_state = np.array([10.0, 10.0, 5.0, 1.0, 0.0], dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.state, self.target_state[:3]]).astype(np.float32)

    def compute_reward(self, prev_state, cur_state):
        prev_dist = np.linalg.norm(prev_state[:3] - self.target_state[:3])
        cur_dist = np.linalg.norm(cur_state[:3]  - self.target_state[:3])
        delta = prev_dist - cur_dist

        reward = delta
        
        reward -= 0.01 # Time penalty
        
        if cur_dist < 0.5:
            reward += 50.0 # Bonus for reaching the target

        return reward

    def step(self, action):
        # Target random movement, get random variation
        x_t, y_t, z_t, v_t, θ_t = self.target_state
        dθ_t = np.random.uniform(-self.delta_theta, self.delta_theta)
        dz_t = np.random.uniform(-self.delta_z,     self.delta_z)
        
        # Apply target movement
        θ_t = (θ_t + dθ_t + np.pi) % (2*np.pi) - np.pi
        z_t += dz_t
        x_t += v_t * np.cos(θ_t) * self.dt
        y_t += v_t * np.sin(θ_t) * self.dt
        prev_target = self.target_state.copy()
        self.target_state = np.array([x_t, y_t, z_t, v_t, θ_t], dtype=np.float32)
        
        prev = self.state.copy()
        prev_dist = np.linalg.norm(prev[:3] - prev_target[:3])
        target_moved_dist = np.linalg.norm(prev - self.target_state[:3])
        delta_target = prev_dist - target_moved_dist

        # Agent movement
        x, y, z, v, θ = self.state
        dθ, dz = action
        θ = (θ + dθ + np.pi) % (2*np.pi) - np.pi
        z += dz
        x += v * np.cos(θ) * self.dt
        y += v * np.sin(θ) * self.dt
        self.state = np.array([x, y, z, v, θ], dtype=np.float32)
        self.current_step += 1
        
        curr_dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        delta_agent = target_moved_dist - curr_dist

        # Reward and termination
        reward = self.compute_reward(prev, self.state)
        dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        terminated = bool(dist < 0.5)
        truncated = bool(self.current_step >= self.max_steps)
        info = {"distance": float(dist)} # Optional info, useful for logging

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Agent pos: {self.state[:3]}  |  Target pos: {self.target_state[:3]}")
