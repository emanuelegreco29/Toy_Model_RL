import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class PointMassEnv(gym.Env):
    dt = 0.1
    max_steps = 500

    def __init__(self, K_history: int = 1):
        super().__init__()
        # History length for past K target states
        self.K_history = K_history
        self.k_att = 5.0
        self.d_follow = 1.5
        self.use_shaping = True
        self.use_force = True
        self.gamma = 0.99
        # Movement parameters
        self.bounds = np.array([[0, 10.0], [0, 10.0], [0, 10.0]], dtype=np.float32)
        self.initial_speed = 1.0
        self.max_step = 500
        self.tracking_counter = 0
        # Agent control increments
        self.delta_yaw = 0.4
        self.delta_v = 0.1
        self.delta_pitch = 0.4
        self.v_max = 1.5
        self.v_min = 0.1

        # Action space: change in heading and altitude
        low_act = np.array([-self.delta_v, -self.delta_yaw, -self.delta_pitch], dtype=np.float32)
        high_act = np.array([ self.delta_v, self.delta_yaw,  self.delta_pitch], dtype=np.float32)
        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)

        # Observation: agent state (6) + target history (K+1 states of length 3, just x,y,z)
        obs_dim = 6 + 3 * (self.K_history + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        if self.use_force:
            old_dim = self.observation_space.shape[0]
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(old_dim + 3,), dtype=np.float32
            )

        # Pre-compute a straight-line trajectory along +X at constant speed
        start = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        dir_unit = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.target_traj = np.array([
            start + i * self.initial_speed * self.dt * dir_unit
            for i in range(self.max_steps + 1)
        ], dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.totally_behind = 0
        self.tracking_counter = 0
        self.prev_target_vel = np.zeros(3, dtype=np.float32)
        self.target_vel = np.zeros(3, dtype=np.float32)

        # Initialize target history
        self.target_history = deque(maxlen=self.K_history+1)
        first_t = self.target_traj[0]
        for _ in range(self.K_history+1):
            self.target_history.append(first_t)
        self.target_state = first_t.copy()

        self.state = np.array([
            2, 2 , 2,
            self.initial_speed,
            0, 0
        ], dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        st = self.state.copy()                # [x,y,z, v, yaw, pitch]
        hist = np.array(self.target_history)  # shape (K+1,3)
        deltas = np.diff(hist, axis=0)       # (K,3)
        vels   = deltas / self.dt            # (K,3)
        speeds = np.linalg.norm(vels, axis=1)# (K,)
        accs   = np.diff(speeds) / self.dt if len(speeds)>1 else np.zeros(0)
        headings = np.arctan2(deltas[:,1], deltas[:,0])          if len(deltas)>0 else np.zeros(0)
        yaw_rates = np.diff(headings) / self.dt                  if len(headings)>1 else np.zeros(0)

        last_speed    = speeds[-1]    if len(speeds)>0    else 0.0
        last_acc      = accs[-1]      if len(accs)>0      else 0.0
        last_yaw_rate = yaw_rates[-1] if len(yaw_rates)>0 else 0.0

        flat_hist = hist.ravel()  # (3*(K+1),)

        return np.concatenate([
            st,
            flat_hist,
            [last_speed, last_acc, last_yaw_rate]
        ]).astype(np.float32)
    
    def _is_behind(self):
        agent_pos  = self.state[:3]
        target_pos = self.target_history[-1]
        if len(self.target_history) < 2:
            return False
        # Movement is always along +X
        dir_unit = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec = agent_pos - target_pos
        dist = np.linalg.norm(vec)
        if dist > 2.0:
            return False
        proj = np.dot(vec, dir_unit)
        if not (-2.0 <= proj <= 0.0):
            return False
        perp = np.linalg.norm(vec - proj*dir_unit)
        return perp <= 0.35

    def compute_tail_goal(self, target_pos, target_vel):
        # se la velocità è quasi nulla, rimani a fianco
        speed = np.linalg.norm(target_vel)
        if speed < 1e-3:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = target_vel / speed
        # punto dietro il target
        return target_pos - self.d_follow * direction

    def compute_attractive_force(self, agent_pos, goal_pos):
        # F_att = -k_att * (x - x_goal)
        return -self.k_att * (agent_pos - goal_pos)

    def potential(self, pos, tail_goal):
        # U_att = 0.5*k_att * ||pos - goal||^2
        return 0.5 * self.k_att * np.linalg.norm(pos - tail_goal)**2
    
    def compute_reward(self, prev, cur):
        # reward base: -distanza attuale dall'obiettivo di tail        
        tail_goal = self.compute_tail_goal(cur[:3], self.target_vel)
        base = -np.linalg.norm(cur[:3] - tail_goal)

        if not self.use_shaping:
            return float(base)

        # shaping: φ = -U, R' = R + γφ(s') - φ(s)
        prev_goal = self.compute_tail_goal(prev[:3], self.prev_target_vel)
        U_prev = self.potential(prev[:3], prev_goal)
        U_cur  = self.potential(cur[:3],  tail_goal)
        phi_prev = -U_prev
        phi_cur  = -U_cur
        return float(base + self.gamma * phi_cur - phi_prev)
    
    def step(self, action):
        prev_agent = self.state.copy()
        self.prev_target_vel = self.target_vel.copy() # previous target velocity

        # Advance target along the straight-line traj
        next_idx    = min(self.current_step + 1, len(self.target_traj) - 1)
        next_target = self.target_traj[next_idx]
        self.target_history.append(next_target)
        self.target_state = next_target.copy()

        # Update agent
        x, y, z, v, yaw, pitch = self.state
        dv, dyaw, dpitch = action
        self.target_vel = self.target_state[3:6]  # target velocity is the last 3 elements of target state
        v = np.clip(v + dv, self.v_min, self.v_max)
        yaw   = (yaw + dyaw + np.pi) % (2*np.pi) - np.pi
        pitch = np.clip(pitch + dpitch, -np.pi/2, np.pi/2)

        dx = v * np.cos(pitch) * np.cos(yaw) * self.dt
        dy = v * np.cos(pitch) * np.sin(yaw) * self.dt
        dz = v * np.sin(pitch)               * self.dt

        x += dx; y += dy; z += dz
        self.state = np.array([x, y, z, v, yaw, pitch], dtype=np.float32)

        self.current_step += 1

        if self._is_behind():
            self.tracking_counter += 1
            self.totally_behind += 1
        else:
            self.tracking_counter = 0

        reward = self.compute_reward(prev_agent, self.state)

        if self.use_force:
            tail_goal = self.compute_tail_goal(self.state[:3], self.target_vel)
            F = self.compute_attractive_force(self.state[:3], tail_goal)
            norm = np.linalg.norm(F)
            if norm > 0:
                F = F / norm
            obs = np.concatenate([self._get_obs(), F]).astype(np.float32)

        # Check termination
        dist = np.linalg.norm(self.state[:3] - self.target_state[:3])
        truncated  = bool(self.current_step >= self.max_steps)
        info = {"distance": float(dist), "followed": int(self.totally_behind)}

        return self._get_obs(), reward, False, truncated, info

    def render(self, mode='human'):
        print("Agent:", self.state[:3], "Target:", self.target_state[:3])