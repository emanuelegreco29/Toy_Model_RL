import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PointMassEnv(gym.Env):
    """
    Ambiente semplificato per un punto materiale in 3D.
    Stato: [x, y, z, v, theta]
      - (x, y, z): posizione
      - v: velocità scalare
      - theta: orientamento nel piano xy (radianti)
    """
    def __init__(self, render_mode=None):
        super(PointMassEnv, self).__init__()
        self.dt = 0.1
        self.max_steps = 500
        self.current_step = 0

        # Parametri di variazione e limiti
        self.delta_theta = 0.5
        self.delta_z = 0.25

        # Action space continuo: [Δv, Δθ, Δz]
        low  = np.array([-self.delta_theta, -self.delta_z], dtype=np.float32)
        high = np.array([ self.delta_theta,  self.delta_z], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Osservazione: stato relativo [dx, dy, dz, v, theta]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # Stato iniziale: [x, y, z, v, θ]
        self.initial_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Target fisso
        self.target = np.array([10.0, 10.0, 5.0], dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.state = self.initial_state.copy()
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Calcola lo stato relativo: differenza posizione-target
        delta = self.target - self.state[:3]
        v, theta = self.state[3], self.state[4]
        return np.concatenate([delta, [v, theta]).astype(np.float32)

    def compute_reward(self, prev_state, current_state, target):
        curr_pos = current_state[:3]
        prev_pos = prev_state[:3]
        curr_distance = np.linalg.norm(curr_pos - target)
        prev_distance = np.linalg.norm(prev_pos - target)
        improvement = prev_distance - curr_distance
        
        reward = -curr_distance  # Penalizza la distanza dal target
 
        return reward

    def step(self, action):
        x, y, z, v, theta = self.state

        # action = [Δθ, Δz]
        dtheta, dz = action
        theta = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi  # normalizzato in [-π,π]
        z = z + dz

        x = x + v * np.cos(theta) * self.dt
        y = y + v * np.sin(theta) * self.dt

        prev_state = self.state.copy()
        self.state = np.array([x, y, z, v, theta], dtype=np.float32)
        self.current_step += 1

        reward = self.compute_reward(prev_state, self.state, self.target)

        terminated = np.linalg.norm(self.state[:3] - self.target) < 0.5
        truncated = self.current_step >= self.max_steps
        distance = np.linalg.norm(self.state[:3] - self.target)
        info = {
            "distance": distance,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        if self.render_mode == "human":
            print("Distanza dal target:", np.linalg.norm(self.state[:3] - self.target))