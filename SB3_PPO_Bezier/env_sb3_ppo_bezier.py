import gymnasium as gym
from scipy.interpolate import CubicSpline
from gymnasium import spaces
import numpy as np
from collections import deque

# Generate a smooth 3D B-spline trajectory
def generate_bspline_trajectory(bounds: np.ndarray, max_step: float, dt: float,
                                 T: int = 500, K: int = 6) -> np.ndarray:
    waypoints = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(K, 3))
    t_wp = np.linspace(0.0, 1.0, K)

    spline_x = CubicSpline(t_wp, waypoints[:, 0])
    spline_y = CubicSpline(t_wp, waypoints[:, 1])
    spline_z = CubicSpline(t_wp, waypoints[:, 2])

    t_samples = np.linspace(0.0, 1.0, T)
    traj = np.vstack([spline_x(t_samples), spline_y(t_samples), spline_z(t_samples)]).T

    # Scale steps to respect max_step
    deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    if deltas.max() > 0:
        factor = min(1.0, max_step / deltas.max())
        for i in range(1, T):
            traj[i] = traj[i-1] + factor * (traj[i] - traj[i-1])
    return traj

class PointMassEnv(gym.Env):
    dt = 0.1
    max_steps = 500

    def __init__(self, K_history: int = 1):
        super().__init__()
        # History length for past K target states
        self.K_history = K_history
        # Movement parameters
        self.bounds = np.array([[0, 10.0], [0, 10.0], [0, 10.0]], dtype=np.float32)
        self.initial_speed = 1.0
        self.max_step = self.initial_speed * self.dt
        # Agent control increments
        self.delta_theta = 0.5
        self.delta_z = 0.25

        # Action space: change in heading and altitude
        low_act = np.array([-self.delta_theta, -self.delta_z], dtype=np.float32)
        high_act = np.array([ self.delta_theta,  self.delta_z], dtype=np.float32)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        # Observation: agent state (5) + target history (K+1 states of length 3, just x,y,z)
        obs_dim = 5 + 3 * (self.K_history + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        # Agent starts at origin, speed=1, heading=0
        self.state = np.array([0.0, 0.0, 0.0, self.initial_speed, 0.0], dtype=np.float32)
        # Pre-generate a random B-spline trajectory for the target
        self.target_traj = generate_bspline_trajectory(
            bounds=self.bounds,
            max_step=self.max_step,
            dt=self.dt,
            T=self.max_steps,
            K=6
        )
        # Initialize target history deque
        self.target_history = deque(maxlen=self.K_history + 1)
        first_target = self.target_traj[0]
        for _ in range(self.K_history + 1):
            self.target_history.append(first_target)
        # Current target state
        self.target_state = first_target.copy()

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.state] + list(self.target_history)).astype(np.float32)

    def compute_reward(self, prev_agent: np.ndarray, cur_agent: np.ndarray) -> float:
        # Distance improvement towards current target
        cur_target = self.target_history[-1]
        prev_dist = np.linalg.norm(prev_agent[:3] - cur_target[:3])
        cur_dist  = np.linalg.norm(cur_agent[:3]  - cur_target[:3])
        imp = prev_dist - cur_dist
        if imp > 0:
            reward = imp
        else:
            reward = -0.5 * abs(imp)
        # Small time penalty
        reward -= 0.001
        # Bonus for reaching the target
        if cur_dist < 0.5:
            reward += 50.0

        # Predictive bonus
        if len(self.target_history) >= 2:
            prev_target = self.target_history[-2]
            # Predicted next target position
            pred_next = cur_target + (cur_target - prev_target)
            prev_pred_dist = np.linalg.norm(prev_agent[:3] - pred_next[:3])
            cur_pred_dist  = np.linalg.norm(cur_agent[:3]  - pred_next[:3])
            pred_imp = prev_pred_dist - cur_pred_dist
            # Scale the predictive improvement
            reward += 0.1 * pred_imp
        return float(reward)

    def step(self, action):
        # Save previous agent state for reward calculation
        prev_agent = self.state.copy()

        # Advance target along pre-generated trajectory
        next_idx = min(self.current_step + 1, len(self.target_traj) - 1)
        next_target = self.target_traj[next_idx]
        self.target_history.append(next_target)
        self.target_state = next_target.copy()

        # Update agent state given action
        x, y, z, v, theta = self.state
        dtheta, dz = action
        theta = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi
        z += dz
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        self.state = np.array([x, y, z, v, theta], dtype=np.float32)

        self.current_step += 1
        reward = self.compute_reward(prev_agent, self.state)

        # Check termination
        dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        terminated = bool(dist < 0.5)
        truncated  = bool(self.current_step >= self.max_steps)
        info = {"distance": float(dist)}

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        print("Agent:", self.state[:3], "Target:", self.target_state[:3])