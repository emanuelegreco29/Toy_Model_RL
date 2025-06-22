import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.interpolate import CubicSpline


def generate_bspline_trajectory(bounds: np.ndarray, max_step: float, dt: float, T: int = 500, K: int = 6) -> np.ndarray:
    """
    Generate a smooth 3D trajectory of length T using a cubic B-spline through K random waypoints,
    scaled so that the max per-step movement is <= max_step.

    :param bounds: array shape (3,2), min/max for x, y, z: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
    :param max_step: maximum allowed distance per step (e.g., speed * dt)
    :param dt: time delta per step (for reference/scaling)
    :param T: number of timesteps
    :param K: number of control waypoints
    :return: array shape (T,3) of (x,y,z) positions
    """
    # Sample random 3D waypoints within bounds
    waypoints = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(K, 3))
    t_wp = np.linspace(0.0, 1.0, K)

    # Build cubic splines for each coordinate
    spline_x = CubicSpline(t_wp, waypoints[:, 0])
    spline_y = CubicSpline(t_wp, waypoints[:, 1])
    spline_z = CubicSpline(t_wp, waypoints[:, 2])

    # Sample the spline uniformly in parameter t
    t_samples = np.linspace(0.0, 1.0, T)
    traj = np.vstack([spline_x(t_samples), spline_y(t_samples), spline_z(t_samples)]).T  # (T,3)

    # Enforce max per-step distance ≤ max_step
    deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    if deltas.max() > 0:
        factor = min(1.0, max_step / deltas.max())
        for i in range(1, T):
            traj[i] = traj[i-1] + factor * (traj[i] - traj[i-1])

    return traj


class PointMassEnv(gym.Env):
    dt = 0.1
    max_steps = 500

    def __init__(self):
        super().__init__()
        self.current_step = 0

        self.delta_theta = 0.5
        self.delta_z = 0.25
        self.delta_v = 0.5
        self.v_min = 1.0
        self.v_max = 3.0

        low_agent = np.array([-self.delta_theta, -self.delta_z, -self.delta_v], dtype=np.float32)
        high_agent = np.array([ self.delta_theta,  self.delta_z,  self.delta_v], dtype=np.float32)
        agent_space = spaces.Box(low_agent, high_agent, dtype=np.float32)
        target_space = spaces.Box(low=-self.delta_v, high=self.delta_v, shape=(1,), dtype=np.float32)

        # combined action: agent's (dtheta, dz, dv) and target's dv
        self.action_space = spaces.Dict({'agent': agent_space, 'target': target_space})

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

        self.target_trajectory = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0

        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # define region bounds for trajectory generation
        bounds = np.array([
            [self.state[0] - 5.0, self.state[0] + 15.0],   # x in [-5, +15]
            [self.state[1] - 5.0, self.state[1] + 15.0],   # y in [-5, +15]
            [0.0, 10.0]                                   # z in [0, 10]
        ], dtype=np.float32)

        # target speed and max per-step movement
        initial_speed = 1.0
        max_step = initial_speed * self.dt

        # generate and store the full trajectory for the episode
        self.target_trajectory = generate_bspline_trajectory(
            bounds, max_step, self.dt, T=self.max_steps, K=6
        )

        # initialize target_state from first trajectory point
        x0, y0, z0 = self.target_trajectory[0]
        dx, dy, dz = self.target_trajectory[1] - self.target_trajectory[0]
        theta0 = np.arctan2(dy, dx)
        self.target_state = np.array([x0, y0, z0, initial_speed, theta0], dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.state, self.target_state]).astype(np.float32)

    def compute_reward(self, prev_state, cur_state):
        prev_dist = np.linalg.norm(prev_state[:3] - self.target_state[:3])
        cur_dist  = np.linalg.norm(cur_state[:3]  - self.target_state[:3])
        improvement = prev_dist - cur_dist

        if improvement > 0:
            reward = improvement
        else:
            reward = -0.5 * abs(improvement)

        reward -= 0.001

        if cur_dist < 0.5:
            reward += 50.0
        return reward

    def _step_continuous(self, action_agent: np.ndarray, action_target: np.ndarray):
        # unpack agent actions
        dtheta, dz_agent, dv_agent = action_agent
        # unpack target dv
        dv_target = float(action_target)

        # Update target velocity within [v_min, v_max]
        x_prev, y_prev, z_prev, v_t_prev, theta_prev = self.target_state
        v_t = np.clip(v_t_prev + dv_target, self.v_min, self.v_max)

        # move target along precomputed trajectory regardless of v_t
        if self.current_step < self.max_steps - 1:
            next_pos = self.target_trajectory[self.current_step + 1]
        else:
            next_pos = self.target_trajectory[-1]
        dx, dy, dz = next_pos - np.array([x_prev, y_prev, z_prev])
        theta_t = np.arctan2(dy, dx)
        x_t, y_t, z_t = next_pos
        self.target_state = np.array([x_t, y_t, z_t, v_t, theta_t], dtype=np.float32)

        # Update agent state
        x, y, z, v, theta = self.state
        # apply heading change
        theta = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi
        # apply vertical change
        z += dz_agent
        # apply speed change within [v_min, v_max]
        v = np.clip(v + dv_agent, self.v_min, self.v_max)
        # move agent
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        prev_agent = self.state.copy()
        self.state = np.array([x, y, z, v, theta], dtype=np.float32)

        # increment step
        self.current_step += 1

        # compute reward and termination
        reward = self.compute_reward(prev_agent, self.state)
        dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        terminated = dist < 0.5
        truncated  = self.current_step >= self.max_steps
        info = {"distance": float(dist)}

        return self._get_obs(), reward, terminated, truncated, info

    def step(self, action):
        # Expect action as dict: {'agent': np.ndarray(3,), 'target': np.ndarray(1,)}
        return self._step_continuous(action['agent'], action['target'])

    def render(self, mode='human'):
        print("Agent:", self.state[:3], "v=", self.state[3], "Target:", self.target_state[:3], "v=", self.target_state[3])


class DiscretePointMassEnv(PointMassEnv):
    """
    Wrapper that maps a discrete action index to continuous actions for both agent and target.
    """
    def __init__(self, n_theta=11, n_z=11, n_v=5):
        super().__init__()
        self.n_theta = n_theta
        self.n_z = n_z
        self.n_v = n_v
        # bins for agent
        self.theta_bins = np.linspace(-self.delta_theta, self.delta_theta, n_theta)
        self.z_bins     = np.linspace(-self.delta_z,     self.delta_z,     n_z)
        self.v_bins     = np.linspace(-self.delta_v,     self.delta_v,     n_v)
        # bins for target dv (same as agent dv)
        self.target_v_bins = self.v_bins

        # total discrete actions
        self.action_space = spaces.Discrete(n_theta * n_z * n_v * n_v)

    def step(self, action):
        idx = int(action)
        # decode agent indices
        idx_th = idx // (self.n_z * self.n_v * self.n_v)
        idx_z  = (idx // (self.n_v * self.n_v)) % self.n_z
        idx_v  = (idx // self.n_v) % self.n_v
        # decode target dv index
        idx_tv = idx % self.n_v

        dtheta = self.theta_bins[idx_th]
        dz     = self.z_bins[idx_z]
        dv     = self.v_bins[idx_v]
        dv_t   = self.target_v_bins[idx_tv]

        action_agent  = np.array([dtheta, dz, dv], dtype=np.float32)
        action_target = np.array([dv_t], dtype=np.float32)
        return self._step_continuous(action_agent, action_target)
