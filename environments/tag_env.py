import numpy as np
import math
from collections import deque
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

class TagEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, K_history: int = 1):
        super().__init__()
        
        # --- Agents ---
        self.agents = ['Agent 0', 'Agent 1']
        self.possible_agents = list(self.agents)
        
        # --- Time and history ---
        self.dt = 0.1
        self.max_steps = 500
        self.K_history = K_history
        
        # --- Agent controls ---
        self.delta_v = 0.1
        self.delta_yaw = 0.4
        self.delta_pitch = 0.4
        self.v_max = 1.5
        self.v_min = 0.1
        
        # Observation space
        obs_dim = 18
        low = -np.inf * np.ones(obs_dim, dtype=np.float32)
        high =  np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_spaces = {a: spaces.Box(low, high, dtype=np.float32)
                                   for a in self.agents}

        # centralized‐critic
        local_dim  = self.observation_spaces[self.agents[0]].shape[0]
        global_dim = local_dim * len(self.agents)
        self.global_state_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(global_dim,), dtype=np.float32
        )

        self.observation_space = dict(self.observation_spaces)
        self.observation_space['global_state'] = self.global_state_space
        
        # Action space: dv, dyaw, dpitch
        act_low = np.array([-self.delta_v, -self.delta_yaw, -self.delta_pitch], dtype=np.float32)
        act_high= np.array([ self.delta_v,  self.delta_yaw,  self.delta_pitch], dtype=np.float32)
        self.action_spaces = {a: spaces.Box(act_low, act_high, dtype=np.float32) for a in self.agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        
        # Initialize states and history
        self.states = {}
        self.history = {}
        self.tracking_counter = {}
        self.total_behind = {}
        
        # Choose one of four initial configurations uniformly
        configs = ['chaser_adv', 'evader_adv', 'front', 'back']
        idx = np.random.randint(0, len(configs))
        cfg = configs[idx]

        if cfg == 'chaser_adv':
            ep_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = 0.0, 0.0
            ch_pos = ep_pos - np.array([2.0, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = yaw_ep, pitch_ep

        elif cfg == 'evader_adv':
            ch_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = 0.0, 0.0
            ep_pos = ch_pos - np.array([2.0, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = yaw_cp, pitch_cp

        elif cfg == 'front':
            ep_pos = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = 0.0, 0.0
            ch_pos = np.array([ 1.0, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = np.pi, 0.0

        elif cfg == 'back':
            ep_pos = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = np.pi, 0.0
            ch_pos = np.array([ 1.0, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = 0.0, 0.0

        speed = 1.0

        # Assign initial state to each agent
        for agent in self.agents:
            if agent == 'Agent 0':
                pos, yaw, pitch = ch_pos, yaw_cp, pitch_cp
            else:
                pos, yaw, pitch = ep_pos, yaw_ep, pitch_ep

            state = np.array([*pos, speed, yaw, pitch], dtype=np.float32)
            self.states[agent] = state

            # reset history and counters
            dq = deque(maxlen=self.K_history+1)
            for _ in range(self.K_history+1):
                dq.append(pos.copy())
            self.history[agent]          = dq
            self.tracking_counter[agent] = 0
            self.total_behind[agent]     = 0
        
        obs = {agent: self._get_obs(agent) for agent in self.agents}
        global_state = np.concatenate([obs[a] for a in self.agents], axis=0)
        infos = {agent: {'global_state': global_state.copy()} for agent in self.agents}

        return obs, infos

    def _get_obs(self, agent):
        # Own and opponent
        own = self.states[agent]
        other = self.states[self._other(agent)]
        
        # History for direction/velocity
        prev, curr = self.history[self._other(agent)][-2], self.history[self._other(agent)][-1]
        delta = curr - prev
        last_vel = delta / self.dt
        if np.linalg.norm(delta) < 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = delta / np.linalg.norm(delta)
            
        # Relative position and angle
        rel = own[:3] - other[:3]
        dist = np.linalg.norm(rel)
        angle = math.atan2(rel[1], rel[0])
        
        obs = np.concatenate([
            rel,                     # 3
            [dist, angle],           # 2
            own,                     # 6
            last_vel.astype(np.float32), # 3
            direction.astype(np.float32),# 3
            [0] # Needed since obs_dim is 18 in policy.pth
        ], axis=0).astype(np.float32)
        
        return obs

    def step(self, action_dict):
        rewards = {}
        infos = {}
        
        # 1) Update positions & histories
        for ag, act in action_dict.items():
            st = self.states[ag].copy()
            x,y,z,v,yaw,pitch = st
            dv, dyaw, dpitch = act
            v     = np.clip(v + dv, self.v_min, self.v_max)
            yaw   = ((yaw + dyaw + np.pi) % (2*np.pi)) - np.pi
            pitch = np.clip(pitch + dpitch, -np.pi/2, np.pi/2)
            dx = v * np.cos(pitch) * np.cos(yaw) * self.dt
            dy = v * np.cos(pitch) * np.sin(yaw) * self.dt
            dz = v * np.sin(pitch)             * self.dt
            new = np.array([x+dx, y+dy, z+dz, v, yaw, pitch], dtype=np.float32)
            self.states[ag] = new
            # update history of positions
            self.history[ag].append(new[:3].copy())
        self.current_step += 1
        
        # 2) Compute rewards
        done = False
        for ag in self.agents:
            
            # behind tracking
            if self._is_behind(ag):
                self.tracking_counter[ag] += 1
                self.total_behind[ag]   += 1
            else:
                self.tracking_counter[ag] = 0
            
            rewards[ag] = self._compute_reward(ag)

            infos[ag] = {
                'distance': float(np.linalg.norm(self.states[ag][:3] - self.states[self._other(ag)][:3])),
                'followed': int(self.total_behind[ag]),
            }
        
        term = (self.current_step >= self.max_steps)
        dones = {ag: done for ag in self.agents}
        dones['__all__'] = term
        
        obs = {ag: self._get_obs(ag) for ag in self.agents}
        global_state = np.concatenate([obs[a] for a in self.agents], axis=0)
        for ag in self.agents:
            infos[ag]['global_state'] = global_state.copy()

        return obs, rewards, dones, infos

    def _other(self, agent):
        return self.agents[1] if agent == self.agents[0] else self.agents[0]

    def _compute_reward(self, agent):
        st = self.states[agent]
        other = self.states[self._other(agent)]
        
        # distance component
        vec = st[:3] - other[:3]
        d3 = np.linalg.norm(vec) + 1e-8
        f_dist = np.exp(-0.5 * (d3 / 10) ** 1)
        
        # positional and velocity alignment
        ux, uy, uz = vec / d3
        position_vec = np.array([ux, uy, uz])
        
        # agent forward vector
        yaw, pitch = st[4], st[5]
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = math.sin(pitch)
        direction_vec = np.array([dx, dy, dz])
        
        # opponent movement direction
        target_dir = self._get_target_direction(agent)
        cos_vel = np.clip(np.dot(direction_vec, target_dir), -1.0, 1.0)
        cos_pos = np.clip(np.dot(position_vec, target_dir), -1.0, 1.0)
        
        f_head_pos = (1 - cos_pos) / 2
        f_head_vel = (1 + cos_vel) / 2
        
        base_reward = (f_dist * 0.5 + f_head_pos * 0.3 + f_head_vel * 0.2) - 1.0
    
        # 2) penalità di allineamento
        other_dir = self._get_target_direction(self._other(agent))
        cos_align = np.clip(np.dot(direction_vec, other_dir), -1.0, 1.0)
        penalty_align = cos_align # -1 < 1

        w_align = 0.2

        return base_reward - w_align * penalty_align

    def _is_behind(self, agent):
        agent_pos = self.states[agent][:3]
        history = list(self.history[self._other(agent)])
        if len(history) < 2:
            return False
        prev, curr = history[-2], history[-1]
        
        dir_vec = curr - prev
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-6:
            return False
        dir_unit = dir_vec / norm
        
        vec = agent_pos - curr
        dist = np.linalg.norm(vec)
        if dist > 2.0:
            return False
        
        proj = np.dot(vec, dir_unit)
        perp_dist = np.linalg.norm(vec - proj * dir_unit)
        
        return -2.0 <= proj <= 0.0 and perp_dist <= 0.5

    def _get_target_direction(self, agent):
        history = list(self.history[self._other(agent)])
        if len(history) < 2:
            return np.array([1.0, 0.0, 0.0])
        
        prev, curr = history[-2], history[-1]
        direction = curr - prev
        norm = np.linalg.norm(direction)
        
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return direction / norm

    def render(self, mode='human'):
        print("Agent 0:", self.states['Agent 0'][:3], 
              "Agent 1:", self.states['Agent 1'][:3])