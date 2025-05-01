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
        self.delta_z = 0.25
        low_act = np.array([-self.delta_theta, -self.delta_z], dtype=np.float32)
        high_act = np.array([ self.delta_theta,  self.delta_z], dtype=np.float32)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self.reset()
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.state = np.array([0.0,0.0,0.0,1.0,0.0], dtype=np.float32)
        self.target_state = np.array([10.0,10.0,5.0,1.0,0.0], dtype=np.float32)
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.concatenate([self.state, self.target_state]).astype(np.float32)
    
    def compute_reward(self, prev_state, cur_state):
        prev_dist = np.linalg.norm(prev_state[:3] - self.target_state[:3])
        cur_dist  = np.linalg.norm(cur_state[:3]  - self.target_state[:3])
        imp = prev_dist - cur_dist
        
        if imp > 0:
            reward = imp
        else:
            reward = -0.5 * abs(imp)
        
        reward -= 0.001
        
        if cur_dist < 0.5:
            reward += 50.0
            
        return reward
    
    def step(self, action):
        
        # update target randomly
        x_t,y_t,z_t,v_t,theta_t = self.target_state
        dtheta_t = np.random.uniform(-self.delta_theta, self.delta_theta)
        dz_t = np.random.uniform(-self.delta_z,     self.delta_z)
        theta_t = (theta_t + dtheta_t + np.pi) % (2*np.pi) - np.pi
        z_t += dz_t
        x_t += v_t * np.cos(theta_t) * self.dt
        y_t += v_t * np.sin(theta_t) * self.dt
        self.target_state = np.array([x_t,y_t,z_t,v_t,theta_t], dtype=np.float32)
        
        # update agent
        x,y,z,v,theta = self.state
        dtheta, dz = action
        theta = (theta + dtheta + np.pi) % (2*np.pi) - np.pi
        z += dz
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        prev = self.state.copy()
        self.state = np.array([x,y,z,v,theta], dtype=np.float32)
        self.current_step += 1
        reward = self.compute_reward(prev, self.state)
        dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        terminated = dist < 0.5
        truncated  = self.current_step >= self.max_steps
        info = {"distance": float(dist)}
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self, mode='human'):
        print("Agent:", self.state[:3], "Target:", self.target_state[:3])
