import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UCAV(gym.Env):
    """
    Point-mass UCAV environment in 3D with continuous action space:
      - actions: [n_x, n_z, mu]
        * n_x: tangential acceleration factor (in g), in [-2, 2]
        * n_z: normal acceleration factor (in g), in [-4, 4]
        * mu: roll angle, in [−π/3, π/3]
      - observations: [R, R_dot, ATA, AA, HTC, Δh, Δv, z_U, v_U]
        as defined in the paper.
    Dynamics are integrated with simple Euler steps.
    """
    def __init__(self,
                 dt=0.1):
        self.dt = dt
        self.g = 9.81
        self.target_position = np.array([1000.0, 0.0, 100.0], dtype=np.float32)
        self.target_velocity = np.array([200.0, 0.0, 0.0], dtype=np.float32)  # [m/s]

        # Utile per reward per decremento distanza
        self.prev_R = None

        # Continuous action space
        self.action_space = spaces.Box(
            low=np.array([-2*self.g, -4*self.g, -np.pi/3], dtype=np.float32),
            high=np.array([ 2*self.g,  4*self.g,  np.pi/3], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: unbounded but practical limits imposed
        obs_low  = np.array([   0.0,    -np.inf,   0.0,   0.0,   0.0,    -np.inf, -np.inf, 0.0,    0.0   ], dtype=np.float32)
        obs_high = np.array([ np.inf,     np.inf, np.pi,   np.pi,   np.pi,     np.inf,  np.inf,  np.inf, np.inf ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.state = None

        self.reward_params = {
            'R_w':       500.0,    # weapon attack range [m]
            'phi_w':     np.deg2rad(20.0),  # max attack angle φ_w [rad]
            'q_w':       np.deg2rad(20.0),  # max aspect angle q_w [rad]
            'R_min':     50.0,
            'R_max':     2000.0,
            'z_min':     20.0,
            'z_max':     500.0,
            'v_min':     100.0,
            'v_max':     400.0,
            'sigma_ang': 3.0,      # for angle advantage
            'sigma_dist': 600.0,   # for distance advantage
            'sigma_alt': 100.0,    # for altitude advantage
            'omega_ang': 1.0,
            'omega_d':   1.0,
            'omega_h':   1.0,
            'a_thr':     0.2,      # lower advantage threshold
            'b_thr':     0.7       # upper advantage threshold
        }
   
        self.reset()

    def reset(self, seed=None, options=None):
        # posizione iniziale e stato di volo
        x0, y0, z0 = 0.0, 0.0, 50.0     # [m]
        v0 = 150.0                      # [m/s]
        gamma0, psi0 = 0.0, 0.0         # [rad]
        self.state = np.array([x0, y0, z0, v0, gamma0, psi0], dtype=np.float32)
        return self._get_obs(), {}
    

    def _get_obs(self):
        # unpack ownship state
        x, y, z, v, gamma, psi = self.state

        # position & velocity vectors
        P_U = np.array([x, y, z], dtype=np.float32)
        v_U = np.array([
            v * np.cos(gamma) * np.cos(psi),
            v * np.cos(gamma) * np.sin(psi),
            v * np.sin(gamma)
        ], dtype=np.float32)
        P_T = self.target_position
        v_T = self.target_velocity

        # line-of-sight vector and distance
        LOS = P_T - P_U
        R   = np.linalg.norm(LOS) + 1e-6

        # range rate
        R_dot = np.dot(LOS, v_T - v_U) / R

        # aspect & antenna train & heading crossing angles
        ATA = np.arccos(np.clip(np.dot(LOS, v_U) / (R * np.linalg.norm(v_U)), -1, 1))
        AA  = np.arccos(np.clip(np.dot(LOS, v_T) / (R * np.linalg.norm(v_T)), -1, 1))
        HTC = np.arccos(np.clip(np.dot(v_U, v_T) / (np.linalg.norm(v_U) * np.linalg.norm(v_T)), -1, 1))

        # altitude & speed differences
        delta_h = P_T[2] - P_U[2]
        delta_v = np.linalg.norm(v_T) - v

        return np.array([R, R_dot, ATA, AA, HTC, delta_h, delta_v, P_U[2], v], dtype=np.float32)

    def step(self, action):
        n_x, n_z, mu = action / np.array([self.g, self.g, 1.0], dtype=np.float32)  # normalize back to multiples of g & rad

        x, y, z, v, gamma, psi = self.state

        # kinematic / dynamic derivatives
        dx    = v * np.cos(gamma) * np.cos(psi)
        dy    = v * np.cos(gamma) * np.sin(psi)
        dz    = v * np.sin(gamma)
        dv    = self.g * (n_x - np.sin(gamma))
        dgamma= (self.g / v) * (n_z * np.cos(mu) - np.cos(gamma))
        dpsi  = (n_z * np.sin(mu)) / (v * np.cos(gamma))

        # Euler integration
        self.state += np.array([dx, dy, dz, dv, dgamma, dpsi], dtype=np.float32) * self.dt
        obs = self._get_obs()

        R = obs[0]
        rew = 0.0
        if self.prev_R is not None:
            rew = self.prev_R - R
        self.prev_R = R

        reward = self.total_reward(obs, self.reward_params) + 0.1 * rew # add small reward for decreasing distance

        return obs, reward, False, False, {}
    
    def compute_final_reward(self, R, ATA, AA, R_w, phi_w, q_w):
        """
        +10 if R≤R_w and ATA<φ_w and AA<q_w
        -10 if R≤R_w and (ATA>π φ_w or AA>π q_w)
        0 otherwise
        """
        if R <= R_w and ATA < phi_w and AA < q_w:
            return 10.0
        if R <= R_w and (ATA > np.pi - phi_w or AA > np.pi - q_w):
            return -10.0
        return 0.0

    def compute_environment_penalty(self, R, R_min, R_max, z_U, z_min, z_max, v_U, v_min, v_max):
        """
        -10 if R<R_min or R>R_max
        -10 if z_U<z_min or z_U>z_max
        -10 if v_U<v_min or v_U>v_max
        0 otherwise
        """
        if R < R_min or R > R_max:
            return -10.0
        if z_U < z_min or z_U > z_max:
            return -10.0
        if v_U < v_min or v_U > v_max:
            return -10.0
        return 0.0

    def advantage_functions(self, ATA, AA, R, R_w, delta_h, 
                            sigma_ang, sigma_dist, sigma_alt):
        """
        angle, distance, altitude advantage f_ang, f_d, f_h.
        """
        # (1) angle advantage
        f_ang = 1.0 - (ATA + AA) / (2.0 * np.pi)

        # (2) distance advantage
        if R <= R_w:
            f_d = 1.0
        else:
            f_d = np.exp(-((R - R_w)**2) / (2.0 * sigma_dist**2))

        # (3) altitude advantage
        Δh = -delta_h  # now Δh = z_U − z_T
        if Δh <= 0:
            f_h = np.exp(-((-Δh)**2) / (2.0 * sigma_alt**2))
        elif 0 < Δh <= sigma_alt:
            f_h = 1.0
        else:
            f_h = np.exp(-((Δh - sigma_alt)**2) / (2.0 * sigma_alt**2))

        return f_ang, f_d, f_h

    def compute_auxiliary_reward(self, f_ang, f_d, f_h,
                                omega_ang, omega_d, omega_h,
                                a_thr, b_thr):
        f_t = omega_ang * f_ang + omega_d * f_d + omega_h * f_h
        if f_t > b_thr:
            return f_t + 5.0
        if a_thr <= f_t <= b_thr:
            return f_t
        return f_t - 6.0

    def total_reward(self, obs, params):
        """
        obs: array([R, R_dot, ATA, AA, HTC, delta_h, delta_v, z_U, v_U])
        params: dict containing all needed thresholds, ranges, sigma's, w's, φ_w, q_w
        """
        R, R_dot, ATA, AA, HTC, delta_h, delta_v, z_U, v_U = obs

        # unpack parameters
        R_w    = params['R_w']
        phi_w  = params['phi_w']
        q_w    = params['q_w']
        R_min  = params['R_min']
        R_max  = params['R_max']
        z_min  = params['z_min']
        z_max  = params['z_max']
        v_min  = params['v_min']
        v_max  = params['v_max']
        sigma_ang  = params['sigma_ang']
        sigma_dist = params['sigma_dist']
        sigma_alt  = params['sigma_alt']
        ω_ang  = params['omega_ang']
        ω_dist = params['omega_d']
        ω_alt  = params['omega_h']
        a_thr  = params['a_thr']
        b_thr  = params['b_thr']

        # compute each component
        r_final = self.compute_final_reward(R, ATA, AA, R_w, phi_w, q_w)
        r_env   = self.compute_environment_penalty(R, R_min, R_max, z_U, z_min, z_max, v_U, v_min, v_max)

        f_ang, f_d, f_h = self.advantage_functions(
            ATA, AA, R, R_w, delta_h,
            sigma_ang, sigma_dist, sigma_alt
        )
        
        r_aux = self.compute_auxiliary_reward(
            f_ang, f_d, f_h,
            ω_ang, ω_dist, ω_alt,
            a_thr, b_thr
        )

        return r_final + r_env + r_aux