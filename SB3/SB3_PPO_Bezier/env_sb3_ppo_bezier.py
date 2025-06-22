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
        self.max_step = 500
        self.tracking_counter = 0
        # Agent control increments
        self.delta_yaw = 0.4
        self.delta_v = 0.1
        self.delta_pitch = 0.4
        self.v_max = 1.2
        self.v_min = 0.2

        # Action space: change in heading and altitude
        low_act = np.array([-self.delta_v, -self.delta_yaw, -self.delta_pitch], dtype=np.float32)
        high_act = np.array([ self.delta_v, self.delta_yaw,  self.delta_pitch], dtype=np.float32)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        # Observation: agent state (6) + target history (K+1 states of length 3, just x,y,z)
        obs_dim = 6 + 3 * (self.K_history + 1) + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.totally_behind = 0
        self.tracking_counter = 0

        # Agent starts at origin, speed=1, yaw=0, pitch=0
        # self.state = np.array([0.0, 0.0, 0.0, self.initial_speed, 0.0, 0.0], dtype=np.float32)

        # Pre-generate a random B-spline trajectory for the target
        self.target_traj = generate_bspline_trajectory(
            bounds=self.bounds,
            max_step=self.max_step,
            dt=self.dt,
            T=self.max_steps,
            K=6 # Number of waypoints
        )
        # Initialize target history deque
        self.target_history = deque(maxlen=self.K_history + 1)
        first_target = self.target_traj[0]
        for _ in range(self.K_history + 1):
            self.target_history.append(first_target)
        # Current target state
        self.target_state = first_target.copy()

        # Start agent behind target
        if len(self.target_traj) > 1:
            p0 = self.target_traj[0]
            p1 = self.target_traj[1]
            dir_vec  = p1 - p0
            dir_unit = dir_vec / np.linalg.norm(dir_vec)
            # agente at distance 2
            init_pos = p0 - 2.0 * dir_unit
            # yaw e pitch corresponding to the target heading
            yaw   = np.arctan2(dir_unit[1], dir_unit[0])
            pitch = np.arcsin(np.clip(dir_unit[2], -1.0, 1.0))
        else:
            init_pos = np.zeros(3, dtype=np.float32)
            yaw = pitch = 0.0

        self.state = np.array([
            init_pos[0], init_pos[1], init_pos[2],
            self.initial_speed,
            yaw, pitch
        ], dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        st = self.state.copy()           # [x,y,z,v,yaw,pitch]

        # history posizioni: shape (K+1,3)
        hist = np.array(self.target_history)

        # calcolo Δ posizioni e velocità: shape (K,3)
        deltas = np.diff(hist, axis=0)
        vels   = deltas / self.dt        # (K,3)
        speeds = np.linalg.norm(vels, axis=1)   # (K,)

        # calcolo accelerazione (lunghezza K-1)
        accs = np.diff(speeds) / self.dt         # (K-1,)

        # calcolo heading e yaw rate (su XY)
        headings = np.arctan2(deltas[:,1], deltas[:,0])           # (K,)
        yaw_rates = np.diff(headings) / self.dt                  # (K-1,)

        # prendo gli ultimi valori se esistono, altrimenti zero
        last_speed    = speeds[-1]    if len(speeds)>0    else 0.0
        last_acc      = accs[-1]      if len(accs)>0      else 0.0
        last_yaw_rate = yaw_rates[-1] if len(yaw_rates)>0 else 0.0

        # flatten della history delle posizioni come prima
        flat_hist = hist.ravel()

        return np.concatenate([
            st,
            flat_hist,
            [last_speed, last_acc, last_yaw_rate]
        ]).astype(np.float32)
    
    def _predict_target(self, history, horizon = 1):
        """
        history: array di forma (k,3) con le ultime k posizioni.
        horizon: passi futuri da predire (default 1).
        Restituisce array (3,) = posizione predetta a t+horizon.
        """
        k = history.shape[0]
        t = np.arange(k).reshape(-1,1)
        A = np.hstack([t, np.ones_like(t)]) # matrix for linear fit

        pred = np.zeros(3, dtype=np.float32)
        t_next = (k-1) + horizon
        for dim in range(3):
            y = history[:, dim]
            (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
            pred[dim] = a * t_next + b
        return pred
    
    def _is_behind(self):
        """
        Determine if the agent is behind the target within a specified cone.

        The function checks if the agent is positioned behind the target,
        relative to the target's movement direction, within a certain 
        distance and angular constraints. It calculates the vector from
        the previous to the current target position to establish the 
        direction of movement. Then, it checks if the agent is within 
        a specified distance from the target and projects the agent's 
        position onto the target's direction vector to determine if 
        it lies behind the target. Additionally, it verifies if the 
        perpendicular distance from the agent to the direction vector 
        is within a specified threshold.
        """
        agent_pos = self.state[:3]
        target_pos = self.target_history[-1]
        # We can calculate only after second timestep, at least
        if len(self.target_history) < 2:
            return False
        prev_target = self.target_history[-2]

        # Target direction vector
        dir_vec = target_pos - prev_target
        if np.linalg.norm(dir_vec) < 1e-6:
            return False
        dir_unit = dir_vec / np.linalg.norm(dir_vec)

        # Agent->Target vector
        vec = agent_pos - target_pos
        dist = np.linalg.norm(vec)
        if dist > 2.0:
            return False

        # Projection along direction
        proj = np.dot(vec, dir_unit)
        # Condition to be behind (inside the cone)
        if not (-2.0 <= proj <= 0.0):
            return False

        # Check perpendicular distance
        perp_dist = np.linalg.norm(vec - proj * dir_unit)
        return perp_dist <= 0.5

    def compute_reward(self, prev_state, state):
        # posizione agente e target
        agent_pos  = state[:3]
        target_pos = self.target_state
        vec        = target_pos - agent_pos
        dist       = np.linalg.norm(vec)
        if dist < 1e-6:
            return 0.0  # reward massima: raw=1→1-1=0

        # velocità agente e target
        v_agent = state[3]
        if len(self.target_history) >= 2:
            v_target = np.linalg.norm((self.target_history[-1]
                                      - self.target_history[-2]) / self.dt)
        else:
            v_target = v_agent
        v_target = max(v_target, 1e-3)

        # --- ORIZZONTALE ----------------------------------------------------
        # errore yaw
        desired_yaw  = np.arctan2(vec[1], vec[0])
        yaw_err      = (state[4] - desired_yaw + np.pi) % (2*np.pi) - np.pi
        heading_scale = np.deg2rad(8.0)      # 8° tolleranza
        # distanza piana
        vec_xy    = vec.copy(); vec_xy[2] = 0
        d_xy      = np.linalg.norm(vec_xy)
        xy_scale  = 3.0                       # 3 m tolleranza
        # reward
        r_yaw     = np.exp(- (yaw_err / heading_scale)**2)
        r_xy_dist = np.exp(- (d_xy   / xy_scale  )**2)
        r_horiz   = r_yaw * r_xy_dist         # in [0,1]

        # --- VERTICALE -----------------------------------------------------
        # errore pitch
        dist_z      = vec[2]
        desired_pitch = np.arcsin(np.clip(dist_z / dist, -1.0, 1.0))
        pitch_err    = state[5] - desired_pitch
        pitch_scale  = np.deg2rad(8.0)
        z_scale      = 3.0
        r_pitch      = np.exp(- (pitch_err / pitch_scale)**2)
        r_z_dist     = np.exp(- (dist_z      / z_scale   )**2)
        r_vert       = r_pitch * r_z_dist    # in [0,1]

        # --- VELOCITÀ -------------------------------------------------------
        v_scale = 0.5
        r_vel   = np.exp(- ((v_agent - v_target) / v_scale)**2)  # in [0,1]

        # --- COMBINAZIONE --------------------------------------------------
        w_h, w_v, w_vl = 1.0, 1.0, 1.0
        raw = (w_h * r_horiz + w_v * r_vert + w_vl * r_vel) / (w_h + w_v + w_vl)
        # sposto in [-1,0]
        return raw - 1.0

    
    def step(self, action):
        # Save previous agent state for reward calculation
        prev_agent = self.state.copy()

        # Advance target along pre-generated trajectory
        next_idx = min(self.current_step + 1, len(self.target_traj) - 1)
        next_target = self.target_traj[next_idx]
        self.target_history.append(next_target)
        self.target_state = next_target.copy()

        # Update agent state given action
        x, y, z, v, yaw, pitch = self.state
        dv, dyaw, dpitch = action

        # Aggiorno velocità e angoli, mantenendoli in range sensato
        v = np.clip(v + dv, self.v_min, self.v_max)
        yaw = (yaw + dyaw + np.pi) % (2*np.pi) - np.pi
        pitch = np.clip(pitch + dpitch, -np.pi/2, np.pi/2)

        # Calcolo gli spostamenti
        dx = v * np.cos(pitch) * np.cos(yaw) * self.dt
        dy = v * np.cos(pitch) * np.sin(yaw) * self.dt
        dz = v * np.sin(pitch)               * self.dt

        # Applico i vincoli dei bounds
        x = x + dx
        y = y + dy
        z = z + dz

        # Nuovo stato
        self.state = np.array([x, y, z, v, yaw, pitch], dtype=np.float32)

        self.current_step += 1

        if(self._is_behind()):
            self.tracking_counter += 1
            self.totally_behind += 1
        else:
            self.tracking_counter = 0
        reward = self.compute_reward(prev_agent, self.state)

        # Check termination
        dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        truncated  = bool(self.current_step >= self.max_steps)
        info = {"distance": float(dist), "followed": int(self.totally_behind)}

        return self._get_obs(), reward, False, truncated, info

    def render(self, mode='human'):
        print("Agent:", self.state[:3], "Target:", self.target_state[:3])