import gymnasium as gym
from scipy.interpolate import CubicSpline
from gymnasium import spaces
import numpy as np
import math
from collections import deque

# Generate a smooth 3D B-spline trajectory

def generate_bspline_trajectory(bounds: np.ndarray, max_step: float, dt: float,
                                 T: int = 500, K: int = 6) -> np.ndarray:
    """
    Genera una traiettoria spline 3D regolare entro "bounds", assicurandosi
    che lo step massimo sia "max_step".
    """
    # Sample K random waypoints in the 3D bounds
    waypoints = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(K, 3))
    t_wp = np.linspace(0.0, 1.0, K)

    # Fit cubic splines for x, y, z
    spline_x = CubicSpline(t_wp, waypoints[:, 0])
    spline_y = CubicSpline(t_wp, waypoints[:, 1])
    spline_z = CubicSpline(t_wp, waypoints[:, 2])

    # Sample densely
    t_samples = np.linspace(0.0, 1.0, T)
    traj = np.vstack([spline_x(t_samples), spline_y(t_samples), spline_z(t_samples)]).T

    # Scale steps to respect max_step
    deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    if deltas.max() > 0:
        factor = min(1.0, max_step / deltas.max())
        # Scale each segment
        traj[1:] = traj[:-1] + (traj[1:] - traj[:-1]) * factor
    return traj


class PointMassEnv(gym.Env):
    """
    Ambiente Gym per un point-mass che deve seguire un target generato
    da una traiettoria spline 3D. Il reward è un cono 3D progressivo (WEZ)
    calcolato on-the-fly punto per punto.
    """
    dt = 0.1
    max_steps = 500

    def __init__(self, K_history: int = 5):
        super().__init__()
        # --- Storico target ---
        self.K_history = K_history

        self.bounds = np.array([[0, 10.0], [0, 10.0], [0, 10.0]], dtype=np.float32) # These are target bounds
        self.initial_speed = 1.0
        self.max_step = 0.5

        # --- Controlli agent ---
        self.delta_yaw = 0.4
        self.delta_v = 0.1
        self.delta_pitch = 0.4
        self.v_max = 1.0
        self.v_min = 0.1

        # --- Azione e osservazione ---
        low_act = np.array([-self.delta_v, -self.delta_yaw, -self.delta_pitch], dtype=np.float32)
        high_act = np.array([ self.delta_v,  self.delta_yaw,  self.delta_pitch], dtype=np.float32)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        obs_dim = 6 + 3*(self.K_history+1) + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Variables for heatmap
        self.tail_offset    = 0.5    # distanza dietro al target
        self.max_reward     = 1.0
        self.min_reward     = 0.0
        self.heatmap_method = 'gaussian'
        self.sigma_dist     = 15.0    # larghezza gaussiana su distanza
        self.sigma_z        = 6.0    # larghezza gaussiana su quota
        self.sigma_heading  = np.deg2rad(60.0) # gradi in radianti, scarto accettabile per l'heading

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.totally_behind = 0
        self.tracking_counter = 0

        self.target_traj = generate_bspline_trajectory(
            bounds=self.bounds,
            max_step=self.max_step,
            dt=self.dt,
            T=self.max_steps,
            K=6
        )

        # Inizializza deque storico
        self.target_history = deque(maxlen=self.K_history+1)
        first = self.target_traj[0].copy()
        for _ in range(self.K_history+1):
            self.target_history.append(first)
        self.target_state = first.copy()

        # Posiziona agent dietro target
        if len(self.target_traj) > 1:
            p0, p1 = self.target_traj[0], self.target_traj[1]
            dir_vec = p1 - p0
            dir_unit = dir_vec / (np.linalg.norm(dir_vec)+1e-8)
            init_pos = p0 - 2.0 * dir_unit
            yaw   = np.arctan2(dir_unit[1], dir_unit[0])
            pitch = np.arcsin(np.clip(dir_unit[2], -1.0, 1.0))
        else:
            init_pos = np.zeros(3, dtype=np.float32)
            yaw = pitch = 0.0

        self.state = np.array([*init_pos, self.initial_speed, yaw, pitch], dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        st = self.state.copy()
        hist = np.array(self.target_history)
        deltas = np.diff(hist, axis=0)
        vels = deltas / self.dt
        speeds = np.linalg.norm(vels, axis=1)
        accs = np.diff(speeds)/self.dt if len(speeds)>1 else np.zeros(1)
        headings = np.arctan2(deltas[:,1], deltas[:,0])
        yaw_rates = np.diff(headings)/self.dt if len(headings)>1 else np.zeros(1)

        last_speed = speeds[-1] if len(speeds)>0 else 0.0
        last_acc   = accs[-1]   if len(accs)>0   else 0.0
        last_yaw   = yaw_rates[-1] if len(yaw_rates)>0 else 0.0

        flat_hist = hist.ravel()
        return np.concatenate([st, flat_hist, [last_speed, last_acc, last_yaw]]).astype(np.float32)
    
    def _is_behind(self):
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

    def compute_reward_pointwise(self, agent_pos, target_pos, target_dir):
        # Punto apice del cono (dietro il target)
        tail = target_pos - target_dir * self.tail_offset
        vec = agent_pos - tail
        d3 = np.linalg.norm(vec) + 1e-8

        # Fattore distanza (gaussiano o lineare)
        if self.heatmap_method == 'gaussian':
            f_dist = np.exp(-0.5 * (d3 / self.sigma_dist) ** 2)
        elif self.heatmap_method == 'slow_exp':
            lam = 0.2  # The smaller this value, the slower the decay
            f_dist = np.exp(-lam * d3)
        else:
            max_d = np.linalg.norm(self.bounds[:,1] - self.bounds[:,0])
            f_dist = np.clip(1 - d3 / max_d, 0, 1)

        # Fattore allineamento in quota
        dz = agent_pos[2] - target_pos[2]
        f_z = np.exp(-0.5 * (dz / self.sigma_z) ** 2)

        # Fattore heading (dot product normalizzato)
        ux, uy, uz = vec / d3
        td_x, td_y, td_z = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        # 3) Calcolo del coseno dell'angolo tra i due vettori
        cos_theta = ux * td_x + uy * td_y + uz * td_z
        # Clip per evitare errori numerici fuori dall'intervallo [-1,1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        if(np.linalg.norm(target_dir[:2]) < 1e-6):
            f_head = 1.0
        else:
            f_head = np.exp(-0.5 * (theta / self.sigma_heading) ** 2)

        # Reward combinato, normalizzato fra min_reward e max_reward
        r = f_dist * f_z * f_head
        r_neg = r - 1.0
        return float(np.clip(r_neg, -1.0, 0.0))

    def compute_reward(self, prev_agent, cur_agent) -> float:
        # Calcola direzione target dal delta storico
        if len(self.target_history) > 1:
            prev_t = self.target_history[-2]
            delta = self.target_state - prev_t
            target_dir = delta / (np.linalg.norm(delta)+1e-8)
        else:
            target_dir = np.array([1.0, 0.0, 0.0])
        return self.compute_reward_pointwise(cur_agent[:3], self.target_state, target_dir)

    def step(self, action):
        prev_agent = self.state.copy()
        # Avanza target
        idx = min(self.current_step+1, len(self.target_traj)-1)
        self.target_history.append(self.target_traj[idx].copy())
        self.target_state = self.target_traj[idx].copy()

        # Muove agent
        x, y, z, v, yaw, pitch = self.state
        dv, dyaw, dpitch = action
        v     = np.clip(v + dv, self.v_min, self.v_max)
        yaw   = (yaw   + dyaw   + np.pi) % (2*np.pi) - np.pi
        pitch = np.clip(pitch + dpitch, -np.pi/2, np.pi/2)
        dx = v * np.cos(pitch) * np.cos(yaw) * self.dt
        dy = v * np.cos(pitch) * np.sin(yaw) * self.dt
        dz = v * np.sin(pitch)               * self.dt
        x, y, z = x+dx, y+dy, z+dz
        self.state = np.array([x, y, z, v, yaw, pitch], dtype=np.float32)

        self.current_step += 1
        reward = self.compute_reward(prev_agent, self.state)

        # Tracking
        if self._is_behind():
            self.tracking_counter += 1
            self.totally_behind   += 1
        else:
            self.tracking_counter = 0

        # Terminazione e info
        dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        truncated = bool(self.current_step >= self.max_steps)
        info = {"distance": float(dist), "followed": int(self.totally_behind)}
        return self._get_obs(), reward, False, truncated, info

    def render(self, mode='human'):
        print("Agent:", self.state[:3], "Target:", self.target_state[:3])