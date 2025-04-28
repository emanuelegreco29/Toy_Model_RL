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
        self.delta_v = 0.25
        self.delta_theta = 0.5
        self.delta_z = 0.25
        self.v_max = 2.0

        # Action space continuo: [Δv, Δθ, Δz]
        low  = np.array([-self.delta_v, -self.delta_theta, -self.delta_z], dtype=np.float32)
        high = np.array([ self.delta_v,  self.delta_theta,  self.delta_z], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Osservazione: stato + target + dir_to_target (3)
        # [x, y, z, v, θ, x_t, y_t, z_t, dx, dy, dz]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # Stato iniziale: [x, y, z, v, θ]
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Target fisso
        self.target = np.array([10.0, 10.0, 5.0], dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return self._get_obs(), {}
    
    def _get_obs(self):
        pos = self.state[:3]
        vec = self.target - pos
        dist = np.linalg.norm(vec)
        dir_to = vec/dist if dist>1e-8 else np.zeros(3, dtype=np.float32)
        return np.concatenate([self.state, self.target, dir_to])

    def compute_reward(self, prev_state, current_state, target):
        curr_pos = current_state[:3]
        prev_pos = prev_state[:3]
        curr_distance = np.linalg.norm(curr_pos - target)
        prev_distance = np.linalg.norm(prev_pos - target)
        improvement = prev_distance - curr_distance
        
        if improvement >= 0:
            reward = 10.0 * improvement
        else:
            reward = - 5.0 * abs(improvement)
        
        reward -= 0.01

 
        # Bonus se il target viene raggiunto (ad esempio, distanza < 0.5)
        if curr_distance < 0.5:
            reward += 100.0
 
        return reward

    def step(self, action):
        x, y, z, v, theta = self.state

        # action = [Δv, Δθ, Δz]
        dv, dtheta, dz = action
        v = np.clip(v + dv, 0.0, self.v_max)
        theta = theta + dtheta
        z = z + dz

        x = x + v * np.cos(theta) * self.dt
        y = y + v * np.sin(theta) * self.dt

        prev_state = self.state.copy()
        self.state = np.array([x, y, z, v, theta], dtype=np.float32)
        self.current_step += 1

        reward = self.compute_reward(prev_state, self.state, self.target)

        terminated = False
        truncated = False
        if np.linalg.norm(self.state[:3] - self.target) < 0.5:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.render_mode == "human":
            print("Distanza dal target:", np.linalg.norm(self.state[:3] - self.target))