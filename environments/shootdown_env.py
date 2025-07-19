import numpy as np
import math
from collections import deque
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

class DogfightParallelEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, K_history: int = 1):
        super().__init__()
        
        # --- Agents ---
        self.agents = ['chaser_0', 'evader_0']
        self.possible_agents = list(self.agents)
        
        # --- Time and history ---
        self.dt = 0.1
        self.max_steps = 500
        self.K_history = K_history
        self.prev_dist = {ag: None for ag in self.agents}
        self.alpha_act = 0.2
        
        # --- WEZ & combat ---
        self.wez_length = 2.0
        self.wez_angle = math.radians(30)
        self.wez_cos_threshold = math.cos(self.wez_angle)
        self.hp_decrement = 5
        self.wez_reward = 0.1
        self.wez_step = 0.05
        self.destroy_bonus = 2.0
        self.hp = {a: 100 for a in self.agents} # HP iniziali
        self.prev_hp = {a: self.hp[a] for a in self.agents} # Differenza HP per il reward
        
        # --- Agent controls ---
        self.delta_v = 0.1
        self.delta_yaw = 0.4
        self.delta_pitch = 0.4
        self.v_max = 1.5
        self.v_min = 1.0
        self.evader_v_max = 1.0
        self.evader_v_min = 0.75
        
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
        self.prev_act = {ag: np.zeros(self.action_spaces[ag].shape, dtype=np.float32)
                    for ag in self.agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        
        # Initialize states and history
        self.states = {}
        self.history = {}
        self.hp = {}
        self.wez_steps = {}
        self.tracking_counter = {}
        self.total_behind = {}
        self.destroyed = {}
        
        # Choose one of four initial configurations uniformly
        configs = ['chaser_adv', 'evader_adv', 'front', 'back']
        idx = np.random.randint(0, len(configs))
        cfg = configs[idx]

        if cfg == 'chaser_adv':
            ep_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = 0.0, 0.0
            ch_pos = ep_pos - np.array([0.5, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = yaw_ep, pitch_ep

        elif cfg == 'evader_adv':
            ch_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = 0.0, 0.0
            ep_pos = ch_pos - np.array([0.5, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = yaw_cp, pitch_cp

        elif cfg == 'front':
            ep_pos = np.array([-0.25, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = 0.0, 0.0
            ch_pos = np.array([ 0.25, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = np.pi, 0.0

        elif cfg == 'back':
            ep_pos = np.array([-0.25, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = np.pi, 0.0
            ch_pos = np.array([ 0.25, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = 0.0, 0.0

        #speed = 1.0

        # Assign initial state to each agent
        for agent in self.agents:
            if agent == 'chaser_0':
                pos, yaw, pitch = ch_pos, yaw_cp, pitch_cp
                speed = np.random.uniform(self.v_min, self.v_max)
            else:  # evader_0
                pos, yaw, pitch = ep_pos, yaw_ep, pitch_ep
                speed = np.random.uniform(self.evader_v_min, self.evader_v_max)

            state = np.array([*pos, speed, yaw, pitch], dtype=np.float32)
            self.states[agent] = state

            # reset history, HP and counters
            dq = deque(maxlen=self.K_history+1)
            for _ in range(self.K_history+1):
                dq.append(pos.copy())
            self.history[agent]          = dq
            self.hp[agent]               = 100
            self.prev_hp[agent]          = self.hp[agent]
            self.wez_steps[agent]        = 0
            self.tracking_counter[agent] = 0
            self.total_behind[agent]     = 0
            self.destroyed[agent]        = False
            
        # Initialize previous distances
        for agent in self.agents:
            d0 = np.linalg.norm(self.states[agent][:3] - self.states[self._other(agent)][:3])
            self.prev_dist[agent] = d0
        
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
            [self.hp[self._other(agent)]]   # 1
        ], axis=0).astype(np.float32)
        
        return obs

    def step(self, action_dict):
        rewards = {}
        infos = {}
        
        # 1) Update positions & histories
        for ag, act in action_dict.items():
            st = self.states[ag].copy()
            
            # Smooth the action
            prev = self.prev_act[ag]
            smoothed = self.alpha_act * prev + (1 - self.alpha_act) * act
            self.prev_act[ag] = smoothed.copy()
            x,y,z,v,yaw,pitch = st
            #dv, dyaw, dpitch = act
            dv, dyaw, dpitch = smoothed # Use smoothed action
            
            # Update state
            if ag == 'chaser_0':
                vmin, vmax = self.v_min, self.v_max
            else:  # evader_0
                vmin, vmax = self.evader_v_min, self.evader_v_max
            v     = np.clip(v + dv, vmin, vmax)
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
        
        # 2) Compute rewards, WEZ, HP, dones
        done = False
        for ag in self.agents:
            
            # behind tracking
            if self._is_behind(ag):
                self.tracking_counter[ag] += 1
                self.total_behind[ag]   += 1
            else:
                self.tracking_counter[ag] = 0
                
            # WEZ steps
            if self._in_wez(ag):
                self.wez_steps[ag] += 1
            else:
                self.wez_steps[ag] = 0
            
            # Assegna le ricompense
            if ag == 'chaser_0':
                rewards[ag] = self._compute_reward_chaser(ag)
            else: # evader_0
                rewards[ag] = self._compute_reward_evader(ag)

            infos[ag] = {
                'distance': float(np.linalg.norm(self.states[ag][:3] - self.states[self._other(ag)][:3])),
                'followed': int(self.total_behind[ag]),
                'target_destroyed': self.destroyed[self._other(ag)],
            }

            if self.destroyed[self._other(ag)] or self.current_step >= self.max_steps:
                done = True
                infos[ag]['final_hp'] = int(self.hp[self._other(ag)])
        
        dones = {ag: done for ag in self.agents}
        obs = {ag: self._get_obs(ag) for ag in self.agents}
        global_state = np.concatenate([obs[a] for a in self.agents], axis=0)
        for ag in self.agents:
            infos[ag]['global_state'] = global_state.copy()

        return obs, rewards, dones, infos

    def _other(self, agent):
        return self.agents[1] if agent == self.agents[0] else self.agents[0]

    def _compute_reward_evader(self, agent):
        st = self.states[agent]
        other = self.states[self._other(agent)]
        
        # distance component (per evader voglio aumentare la distanza)
        vec = st[:3] - other[:3]
        d3 = np.linalg.norm(vec) + 1e-8
        f_dist =  1 - np.exp(-0.5 * (d3 / 10) ** 1)
        
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
        
        f_head_pos = (1 + cos_pos) / 2
        f_head_vel = (1 - cos_vel) / 2
        
        base_reward = (f_dist * 0.5 + f_head_pos * 0.3 + f_head_vel * 0.2) - 1.0
        
        # penalità se l'evader è nella WEZ del chaser
        if self._in_wez(self._other(agent)):
            penalty_wez = -self.wez_reward
        else:
            penalty_wez = 0.0

        # penalità se ha perso HP
        prev = self.prev_hp[agent]
        curr = self.hp[agent]
        hp_diff = prev - curr
        self.prev_hp[agent] = curr
        penalty_hp = -self.hp_decrement if hp_diff > 0 else 0.0

        # penalità se viene ucciso
        penalty_kill = -self.destroy_bonus if self.destroyed[agent] else 0.0

        return base_reward + penalty_wez + penalty_hp + penalty_kill
    
    def _compute_reward_chaser(self, agent):
        st = self.states[agent]
        other = self.states[self._other(agent)]
        
        # distance component
        vec = st[:3] - other[:3]
        d3 = np.linalg.norm(vec) + 1e-8
        f_dist = np.exp(-0.5 * (d3 / 10) ** 1)
        
        # positional and velocity alignment
        ux, uy, uz = vec / d3
        position_vec = np.array([ux, uy, uz])
        
        # distanza attuale e precedente
        d_prev = self.prev_dist[agent]
        d_curr = float(np.linalg.norm(self.states[agent][:3] - other[:3]))
        
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
        
        delta = d_prev - d_curr
            
        k = 0.1
        self.prev_dist[agent] = d_curr

        distance_align_rew = base_reward - w_align * penalty_align + k * delta
        
        # 2) WEZ steps
        if self._in_wez(agent):
            wez = self.wez_step
        else:
            wez = 0.0

        # 3) HP and kill bonus
        if self.wez_steps[agent] >= 5 and not self.destroyed[self._other(agent)]:
            # sottrai HP all'avversario
            opp = self._other(agent)
            self.hp[opp] = max(self.hp[opp] - self.hp_decrement, 0)
            shaping = self.wez_reward
            self.wez_steps[agent] = 0

            # se HP <= 0, kill bonus
            if self.hp[opp] <= 0:
                self.destroyed[opp] = True
                kill = self.destroy_bonus
            else:
                kill = 0.0
        else:
            shaping = 0.0
            kill    = 0.0

        return distance_align_rew + wez + shaping + kill

    def _in_wez(self, agent):
        st = self.states[agent]
        other = self.states[self._other(agent)]
        
        vec = other[:3] - st[:3]
        dist = np.linalg.norm(vec)
        if dist > self.wez_length or dist < 1e-8:
            return False
        
        yaw, pitch = st[4], st[5]
        dx = math.cos(pitch) * math.cos(yaw)
        dy = math.cos(pitch) * math.sin(yaw)
        dz = math.sin(pitch)
        direction_vec = np.array([dx, dy, dz])
        
        rel_unit = vec / dist
        return np.dot(rel_unit, direction_vec) >= self.wez_cos_threshold

    def _is_behind(self, agent):
        agent_pos = self.states[agent][:3]
        history = list(self.history[agent])
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
        print("Chaser:", self.states['chaser_0'][:3], 
              "Evader:", self.states['evader_0'][:3])