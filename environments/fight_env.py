import numpy as np
import math
from collections import deque
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

class FightEnv(ParallelEnv):
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
        self.prev_dist = {ag: None for ag in self.agents}
        self.alpha_act = 0.7
        
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
        self.last_config = cfg  # Store last config for infos

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
        infos = {agent: {'global_state': global_state.copy(), 'config': cfg} for agent in self.agents}

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
                
            # WEZ steps
            if self._in_wez(ag):
                self.wez_steps[ag] += 1
            else:
                self.wez_steps[ag] = 0
            
            rewards[ag] = self._compute_reward(ag)

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

    def _compute_reward(self, agent):
        """
        Reward unificata per self-play:
        - Chiusura della distanza (shaping + f_dist)
        - Align/head-on bonus
        - Penalità di allineamento troppo simmetrico
        - Closure-rate positivo quando MI avvicino più velocemente dell’altro
        - WEZ shaping continuo + WEZ-step/kill step
        - Penalità HP e kill/death
        """
        other = self._other(agent)
        st, ot = self.states[agent], self.states[other]
        
        pos_a = st[:3]; pos_o = ot[:3]
        vec   = pos_a - pos_o
        d_curr = np.linalg.norm(vec) + 1e-8
        d_prev = self.prev_dist[agent]
        delta  = d_prev - d_curr
        self.prev_dist[agent] = d_curr
        
        los_unit = vec / d_curr             # line-of-sight unit vector
        
        # --- Distance shaping + head alignment ---
        f_dist = np.exp(-0.5 * (d_curr / 10)**1)
        
        # head-on: allineamento e orientamento
        yaw, pitch = st[4], st[5]
        dir_a = np.array([math.cos(pitch)*math.cos(yaw),
                          math.cos(pitch)*math.sin(yaw),
                          math.sin(pitch)])
        target_dir = self._get_target_direction(agent)
        cos_pos = np.clip(np.dot(los_unit,      target_dir), -1.0, 1.0)
        cos_vel = np.clip(np.dot(dir_a,         target_dir), -1.0, 1.0)
        f_head_pos = (1 - cos_pos) / 2
        f_head_vel = (1 + cos_vel) / 2
        
        w_dist, w_pos, w_vel, offset = 0.5, 0.3, 0.2, 1.0
        base_reward = (f_dist * w_dist
                     + f_head_pos * w_pos
                     + f_head_vel * w_vel)
        base_reward -= offset
        
        # --- Penalità di troppo allineamento (per evitare tail-chase statico) ---
        other_dir  = self._get_target_direction(other)
        cos_align  = np.clip(np.dot(dir_a, other_dir), -1.0, 1.0)
        w_align    = 0.2
        penalty_align =  w_align * cos_align
        
        # --- Closure-rate (velocità relativa lungo il LOS) ---
        k = 0.5
        pe, ce = self.history[agent][-2], self.history[agent][-1]
        po, co = self.history[other][-2], self.history[other][-1]
        vel_a = (ce - pe) / self.dt
        vel_o = (co - po) / self.dt
        # >0 se MI avvicino più velocemente di quanto l'altro si avvicini a me
        closure_rate = np.dot(vel_a - vel_o, los_unit)
        alpha = 0.5
        
        # --- WEZ shaping + WEZ-step/kill ---
        # penalty proporzionale a quanto entro nella WEZ dell'altro
        d_inside = max(0.0, self.wez_length - d_curr)
        gamma_wez = 0.3
        penalty_wez = - gamma_wez * d_inside
        
        if self._in_wez(agent):
            wez_step = self.wez_step
        else:
            wez_step = 0.0
        
        kill_reward = 0.0
        if self.wez_steps[agent] >= 5 and not self.destroyed[other]:
            self.hp[other] = max(self.hp[other] - self.hp_decrement, 0)
            kill_reward = self.wez_reward
            self.wez_steps[agent] = 0
            if self.hp[other] <= 0:
                self.destroyed[other] = True
                kill_reward += self.destroy_bonus
        
        # --- Penalità HP subiti e death penalty ---
        prev_hp = self.prev_hp[agent]
        curr_hp = self.hp[agent]
        hp_diff = prev_hp - curr_hp
        self.prev_hp[agent] = curr_hp
        penalty_hp    = -self.hp_decrement if hp_diff > 0 else 0.0
        penalty_death = -self.destroy_bonus if self.destroyed[agent] else 0.0
        
        reward = (
            base_reward
          - penalty_align
          + k * delta
          - alpha * closure_rate
          + penalty_wez
          + wez_step
          + kill_reward
          + penalty_hp
          + penalty_death
        )
        return reward

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