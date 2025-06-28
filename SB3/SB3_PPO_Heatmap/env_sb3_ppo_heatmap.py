import gymnasium as gym
from scipy.interpolate import CubicSpline
from gymnasium import spaces
import numpy as np
import math
from matplotlib import pyplot as plt
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

    def __init__(self, K_history: int = 1):
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
        self.v_max = 1.5
        self.v_min = 0.1

        # --- Azione e osservazione ---
        low_act = np.array([-self.delta_v, -self.delta_yaw, -self.delta_pitch], dtype=np.float32)
        high_act = np.array([ self.delta_v,  self.delta_yaw,  self.delta_pitch], dtype=np.float32)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        obs_dim = 17
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

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
        self.state = np.array([0, 0, 0, self.initial_speed, 0.0, 0.0], dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):

        st = self.state.copy() # Posizione, velocità, orientamento dell'agente
        target_pos = self.target_state # Posizione attuale del target

        # ricavo delta tra ultime due posizioni del target
        prev, curr = self.target_history
        delta = curr - prev

        # velocità e direzione normalizzata
        last_vel = delta / self.dt
        direction = delta / (np.linalg.norm(delta) + 1e-8)

        rel_pos = st[:3] - target_pos
        dist = np.linalg.norm(rel_pos)
        angle = np.arctan2(rel_pos[1], rel_pos[0])

        return np.concatenate([
            rel_pos,               # 3
            [dist, angle],         # 2
            st,                    # 6
            last_vel,              # 3
            direction              # 3
        ]).astype(np.float32)
    
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

    def _get_target_direction(self):
        """
        Calcola la direzione del target basata sulla sua velocità/movimento.
        Ritorna la direzione normalizzata del movimento del target.
        """
        if len(self.target_history) < 2:
            # Default direction se non abbiamo abbastanza storia
            return np.array([1.0, 0.0, 0.0])
        
        current_target = self.target_history[-1]
        previous_target = self.target_history[-2]
        
        direction = current_target - previous_target
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-8:
            # Se il target è fermo, usa l'ultima direzione valida o default
            return np.array([1.0, 0.0, 0.0])
        
        return direction / direction_norm
    
    def compute_reward(self):
        agent_pos = self.state[:3]
        yaw, pitch = self.state[4:6]
        target_pos = self.target_state
        target_dir = self._get_target_direction()
        
        vec = agent_pos - target_pos
        d3 = np.linalg.norm(vec) + 1e-8
        f_dist = np.exp(-0.5 * (d3/10) ** 1)

        ux, uy, uz = vec / d3
        position_vec = np.array([ux, uy, uz])
        
        dx = np.cos(pitch) * np.cos(yaw)
        dy = np.cos(pitch) * np.sin(yaw)
        dz = np.sin(pitch)
        direction_vec = np.array([dx, dy, dz])
        
        cos_vel = np.clip(np.dot(direction_vec, target_dir), -1.0, 1.0)
        cos_pos = np.clip(np.dot(position_vec, target_dir), -1.0, 1.0)
        
        f_head_pos = (1 - cos_pos) / 2
        f_head_vel = (1 + cos_vel) / 2

        return (f_dist * 0.5 + f_head_pos * 0.3 + f_head_vel * 0.2) - 1.0

    def step(self, action):
        # Avanza target
        idx = min(self.current_step+1, len(self.target_traj)-1)
        self.target_history.append(self.target_traj[idx].copy())
        self.target_state = self.target_traj[idx].copy()

        # Muove agent (controllo in velocità, yaw e pitch)
        x, y, z, v, yaw, pitch = self.state
        dv, dyaw, dpitch = action
        v     = np.clip(v + dv, self.v_min, self.v_max)
        yaw   = (yaw   + dyaw   + np.pi) % (2*np.pi) - np.pi # Normalizzato tra 180 e -180 gradi
        pitch = np.clip(pitch + dpitch, -np.pi/2, np.pi/2) # Normalizzato tra 90 e -90 gradi
        dx = v * np.cos(pitch) * np.cos(yaw) * self.dt
        dy = v * np.cos(pitch) * np.sin(yaw) * self.dt
        dz = v * np.sin(pitch)               * self.dt

        x, y, z = x+dx, y+dy, z+dz
        self.state = np.array([x, y, z, v, yaw, pitch], dtype=np.float32)

        self.current_step += 1
        reward = self.compute_reward()

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