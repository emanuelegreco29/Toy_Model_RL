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
        #self.tail_offset    = 0.5    # distanza dietro al target
        #self.max_reward     = 1.0
        #self.min_reward     = 0.0
        #self.heatmap_method = 'gaussian'
        #self.sigma_dist     = 15.0    # larghezza gaussiana su distanza
        #self.sigma_z        = 6.0    # larghezza gaussiana su quota
        #self.sigma_heading  = np.deg2rad(60.0) # gradi in radianti, scarto accettabile per l'heading

        # parametri per il cono e i decadimenti
        self.cone_height        = 3.0    # H: altezza del cono dietro il target
        self.cone_radius        = 0.5    # R: raggio di base del cono
        self.behind_decay_range = 10.0   # distanza massima per decadimento dietro
        self.front_decay_range  =  5.0   # distanza massima per decadimento davanti

        # scala reward in [-1,0]
        self.max_reward =  0.0
        self.min_reward = -1.0

        # Parametri per la reward function del cono
        self.exclusion_radius = 0.8  # Raggio di esclusione davanti al target
        self.max_distance = 15.0     # Distanza massima per il calcolo della reward

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

    def cone_reward_function(self, agent_pos, target_pos, cone_direction):
        """
        Calcola la reward per un agent in un environment 3D con un cono di reward.
        
        Args:
            agent_pos: array-like, posizione dell'agent [x, y, z]
            target_pos: array-like, posizione del target [x, y, z]
            cone_direction: array-like, direzione del cono (direzione del movimento del target)
        
        Returns:
            float: reward tra -1 e 0
        """
        agent_pos = np.array(agent_pos)
        target_pos = np.array(target_pos)
        cone_direction = np.array(cone_direction)
        cone_direction = cone_direction / (np.linalg.norm(cone_direction) + 1e-8)
        
        # Vettore dal target all'agent
        target_to_agent = agent_pos - target_pos
        distance_to_target = np.linalg.norm(target_to_agent)
        
        # Zona di esclusione: se l'agent è troppo vicino al target nella direzione opposta al cono
        opposite_direction = -cone_direction
        if distance_to_target > 1e-8:
            direction_to_agent = target_to_agent / distance_to_target
            # Controlla se l'agent è nella zona di esclusione (davanti al target)
            dot_product = np.dot(direction_to_agent, opposite_direction)
            if dot_product > 0.7 and distance_to_target < self.exclusion_radius:  # Circa 45 gradi
                return self.min_reward
        
        # Proiezione del vettore target-agent sulla direzione del cono
        projection_length = np.dot(target_to_agent, cone_direction)
        
        # Se l'agent è nella direzione opposta al cono o troppo lontano
        if projection_length <= 0 or projection_length > self.cone_height:
            # Reward basata sulla distanza
            normalized_distance = min(distance_to_target / self.max_distance, 1.0)
            return self.min_reward * normalized_distance
        
        # Calcola la distanza perpendicolare dall'asse del cono
        projection_point = target_pos + projection_length * cone_direction
        perpendicular_distance = np.linalg.norm(agent_pos - projection_point)
        
        # Raggio del cono a questa altezza
        cone_radius_at_height = self.cone_radius * (projection_length / self.cone_height)
        
        if perpendicular_distance <= cone_radius_at_height:
            # All'interno del cono: reward massima che scala verso il centro
            reward_factor = 1.0 - (perpendicular_distance / cone_radius_at_height)
            # Reward che va da circa -0.1 (bordo del cono) a 0 (centro del cono)
            return self.max_reward - 0.1 * (1.0 - reward_factor)
        else:
            # Fuori dal cono: reward basata sulla distanza dal cono e dal target
            distance_from_cone = perpendicular_distance - cone_radius_at_height
            total_distance = distance_to_target + distance_from_cone
            normalized_distance = min(total_distance / self.max_distance, 1.0)
            return self.min_reward * normalized_distance

    def compute_reward(self, prev_agent_state, current_agent_state):
        """
        Calcola la reward usando la funzione del cono 3D.
        """
        # Posizione corrente dell'agent
        agent_pos = current_agent_state[:3]
        
        # Posizione corrente del target
        target_pos = self.target_history[-1]
        
        # Direzione del target basata sul suo movimento
        target_direction = self._get_target_direction()
        
        # Calcola la reward usando la funzione del cono
        reward = self.cone_reward_function(agent_pos, target_pos, target_direction)
        
        return float(reward)
    
    def visualize_reward_function_2d(self, target_pos=None, target_direction=None, z_slice=0, x_range=(-5, 5), y_range=(-5, 5)):
        """
        Visualizza la reward function in 2D (slice a z costante)
        """
        if target_pos is None:
            target_pos = self.target_history[-1] if len(self.target_history) > 0 else [0, 0, 0]
        if target_direction is None:
            target_direction = self._get_target_direction()
        
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        y_vals = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        rewards = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                agent_pos = [X[i, j], Y[i, j], z_slice]
                rewards[i, j] = self.cone_reward_function(agent_pos, target_pos, target_direction)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(rewards, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                       origin='lower', cmap='viridis')
        plt.colorbar(im, label='Reward')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Reward Heatmap at z={z_slice}')
        plt.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')
        
        # Visualizza la direzione del target
        arrow_length = 1.0
        plt.arrow(target_pos[0], target_pos[1], 
                 target_direction[0] * arrow_length, target_direction[1] * arrow_length,
                 head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.7)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def visualize_reward_function_3d(self, target_pos=None, target_direction=None, n_points=30, 
                                   x_range=(-3, 3), y_range=(-3, 3), z_range=(-3, 1)):
        """
        Visualizza alcuni punti della reward function in 3D
        """
        if target_pos is None:
            target_pos = self.target_history[-1] if len(self.target_history) > 0 else [0, 0, 0]
        if target_direction is None:
            target_direction = self._get_target_direction()
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Crea una griglia di punti
        x_vals = np.linspace(x_range[0], x_range[1], n_points)
        y_vals = np.linspace(y_range[0], y_range[1], n_points)
        z_vals = np.linspace(z_range[0], z_range[1], n_points)
        
        points = []
        rewards = []
        
        # Campiona alcuni punti per la visualizzazione
        for x in x_vals[::3]:
            for y in y_vals[::3]:
                for z in z_vals[::3]:
                    agent_pos = [x, y, z]
                    reward = self.cone_reward_function(agent_pos, target_pos, target_direction)
                    points.append(agent_pos)
                    rewards.append(reward)
        
        points = np.array(points)
        rewards = np.array(rewards)
        
        # Visualizza i punti colorati per reward
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                            c=rewards, cmap='viridis', alpha=0.6, s=20)
        
        # Aggiungi il target
        ax.scatter(*target_pos, color='red', s=200, marker='*', label='Target')
        
        # Visualizza il cono (approssimativo)
        theta = np.linspace(0, 2*np.pi, 20)
        z_cone = np.linspace(0, self.cone_height, 20)
        
        for z in z_cone[::4]:
            r = self.cone_radius * z / self.cone_height
            # Calcola i punti del cerchio nella direzione del cono
            cone_center = np.array(target_pos) + z * np.array(target_direction)
            
            # Crea un sistema di coordinate ortogonale alla direzione del cono
            if abs(target_direction[2]) < 0.9:
                v1 = np.cross(target_direction, [0, 0, 1])
            else:
                v1 = np.cross(target_direction, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(target_direction, v1)
            v2 = v2 / np.linalg.norm(v2)
            
            x_circle = cone_center[0] + r * (np.cos(theta) * v1[0] + np.sin(theta) * v2[0])
            y_circle = cone_center[1] + r * (np.cos(theta) * v1[1] + np.sin(theta) * v2[1])
            z_circle = cone_center[2] + r * (np.cos(theta) * v1[2] + np.sin(theta) * v2[2])
            
            ax.plot(x_circle, y_circle, z_circle, 'r-', alpha=0.3)
        
        # Aggiungi freccia per la direzione del target
        ax.quiver(target_pos[0], target_pos[1], target_pos[2],
                 target_direction[0], target_direction[1], target_direction[2],
                 color='red', arrow_length_ratio=0.1, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Cone Reward Function')
        
        plt.colorbar(scatter, label='Reward')
        plt.legend()
        plt.show()

    def plot_current_reward_state(self):
        """
        Plotta la situazione attuale dell'environment con agent, target e reward heatmap
        """
        if len(self.target_history) < 2:
            print("Non abbastanza dati per plottare")
            return
        
        agent_pos = self.state[:3]
        target_pos = self.target_history[-1]
        
        # Plot 2D slice alla quota dell'agent
        z_slice = agent_pos[2]
        
        # Centra la vista attorno al target
        x_center, y_center = target_pos[0], target_pos[1]
        x_range = (x_center - 5, x_center + 5)
        y_range = (y_center - 5, y_center + 5)
        
        self.visualize_reward_function_2d(target_pos, None, z_slice, x_range, y_range)
        
        # Aggiungi la posizione dell'agent
        plt.plot(agent_pos[0], agent_pos[1], 'bo', markersize=10, label='Agent')
        plt.legend()
        
        # Calcola e mostra la reward attuale
        current_reward = self.cone_reward_function(agent_pos, target_pos, self._get_target_direction())
        plt.title(f'Current Reward: {current_reward:.3f} at z={z_slice:.1f}')
        

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