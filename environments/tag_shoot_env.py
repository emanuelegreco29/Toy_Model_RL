import numpy as np
import math
from collections import deque
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

class TagShootEnv(ParallelEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        
        # --- Agents ---
        self.agents = ['Agent 0', 'Agent 1']
        self.possible_agents = list(self.agents)
        
        # --- Environment parameters ---
        self.dt = 0.1
        self.max_steps = 500
        self.prev_dist = {ag: None for ag in self.agents}
        self.alpha_act = 0.2
        
        # --- WEZ Parameters ---
        self.wez_length = 5.0
        self.wez_cos_threshold = float(np.cos(np.radians(30.0)))
        self.lock_required = 10 # For how many steps the agent needs to keep enemy in WEZ in order to shoot
        self.cooldown_frames = 5 # For how many steps the agent needs to wait before shooting again
        self.lock_counter = {ag: 0 for ag in self.agents}
        self.cooldown = {ag: 0 for ag in self.agents}

        # --- Agent controls ---
        self.delta_v = 0.15
        self.yaw_rate_max = 1.2             # [rad/s] yaw rate limit
        self.pitch_rate_max = 1.0           # [rad/s] pitch rate limit
        self.pitch_abs_max = math.radians(60.0)  # pitch limit
        self.v_max = 1.5
        self.v_min = 1.0

        # Observation space
        obs_dim = 16
        low = -np.inf * np.ones(obs_dim, dtype=np.float32)
        high =  np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_spaces = {a: spaces.Box(low, high, dtype=np.float32)
                                   for a in self.agents}

        # Global state space for centralized critic
        local_dim = self.observation_spaces[self.agents[0]].shape[0]
        global_dim = local_dim * len(self.agents)
        self.global_state_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(global_dim,), dtype=np.float32
        )

        self.observation_space = dict(self.observation_spaces)
        self.observation_space['global_state'] = self.global_state_space
        
        # Action space
        act_low = np.array([-self.delta_v, -self.yaw_rate_max, -self.pitch_rate_max], dtype=np.float32)
        act_high= np.array([ self.delta_v,  self.yaw_rate_max,  self.pitch_rate_max], dtype=np.float32)
        self.action_spaces = {a: spaces.Box(act_low, act_high, dtype=np.float32) for a in self.agents}
        self.prev_act = {ag: np.zeros(self.action_spaces[ag].shape, dtype=np.float32) for ag in self.agents}

        # Tracking variables for reward computation
        self.prev_wez_status = {ag: False for ag in self.agents}
        self.episode_hits = {ag: 0 for ag in self.agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        
        # Initialize states and buffers
        self.states = {}
        self.hp = {}
        self.lock_counter = {ag: 0 for ag in self.agents}
        self.cooldown = {ag: 0 for ag in self.agents}
        self.total_lock = {ag: 0 for ag in self.agents}
        self.prev_act = {ag: np.zeros_like(self.prev_act[ag], dtype=np.float32) for ag in self.agents}
        self.prev_wez_status = {ag: False for ag in self.agents}
        self.episode_hits = {ag: 0 for ag in self.agents}

        # Various initial configurations
        configs = ['chaser_adv', 'evader_adv', 'front', 'back', 'diagonal', 'vertical']
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
            ch_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = np.pi, 0.0

        elif cfg == 'back':
            ep_pos = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            yaw_ep, pitch_ep = np.pi, 0.0
            ch_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            yaw_cp, pitch_cp = 0.0, 0.0
            
        elif cfg == 'diagonal':
            ep_pos = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
            yaw_ep, pitch_ep = np.pi/4, np.pi/6
            ch_pos = np.array([1.0, 1.0, -1.0], dtype=np.float32)
            yaw_cp, pitch_cp = -3*np.pi/4, -np.pi/6
            
        elif cfg == 'vertical':
            ep_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            yaw_ep, pitch_ep = 0.0, -np.pi/3
            ch_pos = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            yaw_cp, pitch_cp = 0.0, np.pi/3

        # Randomize initial speeds
        speed = 1.0
        
        # Assign initial state to each agent
        for i, agent in enumerate(self.agents):
            if agent == 'Agent 0':
                pos, yaw, pitch = ch_pos, yaw_cp, pitch_cp
            else:
                pos, yaw, pitch = ep_pos, yaw_ep, pitch_ep

            state = np.array([*pos, speed, yaw, pitch], dtype=np.float32)
            self.states[agent] = state

            # Reset HP
            self.hp[agent] = 100
            
        # Initialize previous distances
        for agent in self.agents:
            d0 = np.linalg.norm(self.states[agent][:3] - self.states[self._other(agent)][:3])
            self.prev_dist[agent] = d0
        
        obs = {agent: self._get_obs(agent) for agent in self.agents}
        global_state = np.concatenate([obs[a] for a in self.agents], axis=0)
        infos = {}
        for ag in self.agents:
            other = self._other(ag)
            dist = float(np.linalg.norm(self.states[ag][:3] - self.states[other][:3]))
            infos[ag] = {
                'distance': dist,
                'Total Lock': int(self.total_lock[ag]),
                'Agent 0 HP': int(self.hp['Agent 0']),
                'Agent 1 HP': int(self.hp['Agent 1']),
                'global_state': global_state.copy(),
            }

        return obs, infos

    def _get_obs(self, agent):
        """
        Feature restituite:
        0-2:  los (x,y,z)                        [unit]
        3:    distanza normalizzata (d / (4*WEZ)) in [0,1] tagliata a 1
        4:    cos_point = nose_i · los           in [-1,1]
        5:    cos_tail  = nose_j · los           in [-1,1]
        6:    v_close (closure lungo LOS)        >0 se si chiude
        7:    I_wez (0/1)
        8:    v_i (velocità propria)
        9-10: yaw_i, pitch_i
        11:   v_j (velocità avversario)
        12:   enemy_hp (normalizzato su 100)
        13:   aspect_to_me = nose_j · (-los)     in [-1,1]
        14:   my_lock_progress  in [0,1]
        15:   enemy_lock_progress in [0,1]
        """
        other = self._other(agent)

        # Geometria primaria
        los, d = self._los(agent, other)          # unitario, distanza
        ori_i  = self._orientation(agent)
        ori_j  = self._orientation(other)

        # Closure: positivo se la distanza si riduce
        v_close = float(self._closure(agent, other))

        # Indicatori tattici
        I_wez = 1.0 if self._in_wez(agent) else 0.0
        enemy_hp = float(self.hp[other] / 100.0)

        # Puntamento e "tail"
        cos_point = float(np.clip(np.dot(ori_i['nose'], los), -1.0, 1.0))
        cos_tail  = float(np.clip(np.dot(ori_j['nose'], los), -1.0, 1.0))

        # Aspect dell'avversario verso di noi (n_j con -los)
        aspect_to_me = float(np.clip(np.dot(ori_j['nose'], -los), -1.0, 1.0))

        # Progressi lock normalizzati
        my_lock_progress    = float(self.lock_counter[agent] / max(1, self.lock_required))
        enemy_lock_progress = float(self.lock_counter[other]  / max(1, self.lock_required))

        # Distanza normalizzata su scala 4*WEZ
        d_norm = float(min(d / max(1e-6, 4.0 * self.wez_length), 1.0))

        obs = np.array([
            los[0], los[1], los[2],           # 0-2: LOS
            d_norm,                           # 3:   distanza normalizzata
            cos_point,                        # 4
            cos_tail,                         # 5
            v_close,                          # 6:   closure (>0 se chiudiamo)
            I_wez,                            # 7
            ori_i['speed'],                   # 8:   v_i
            ori_i['yaw'], ori_i['pitch'],     # 9-10: yaw_i, pitch_i
            ori_j['speed'],                   # 11:  v_j
            enemy_hp,                         # 12
            aspect_to_me,                     # 13
            my_lock_progress,                 # 14
            enemy_lock_progress               # 15
        ], dtype=np.float32)

        return obs

    def step(self, action_dict):
        rewards = {ag: 0.0 for ag in self.agents}
        infos = {ag: {} for ag in self.agents}

        # Update agents states
        for ag, act in action_dict.items():
            st = self.states[ag].copy()

            # Smooth incremental action
            prev = self.prev_act[ag]
            smoothed = self.alpha_act * prev + (1.0 - self.alpha_act) * np.asarray(act, dtype=np.float32)
            self.prev_act[ag] = smoothed.copy()

            x, y, z, v, yaw, pitch = st
            dv, yaw_rate, pitch_rate = smoothed

            # Compute the changes in state
            v_min, v_max = float(self.v_min), float(self.v_max)
            v = float(np.clip(v + dv, v_min, v_max))

            yaw = float(((yaw + float(np.clip(yaw_rate, -self.yaw_rate_max, self.yaw_rate_max)) * self.dt + np.pi) % (2.0 * np.pi)) - np.pi)

            pitch = float(np.clip(
                pitch + float(np.clip(pitch_rate, -self.pitch_rate_max, self.pitch_rate_max)) * self.dt,
                -self.pitch_abs_max, self.pitch_abs_max
            ))

            # Update state
            dx = v * math.cos(pitch) * math.cos(yaw) * self.dt
            dy = v * math.cos(pitch) * math.sin(yaw) * self.dt
            dz = v * math.sin(pitch) * self.dt

            new = np.array([x + dx, y + dy, z + dz, v, yaw, pitch], dtype=np.float32)
            self.states[ag] = new

        # Advance time
        self.current_step += 1

        # General reward assignment, for hunting
        for ag in self.agents:
            rewards[ag] = float(self._compute_reward(ag))

        # Shooting
        hits = []

        for ag in self.agents:
            prev_wez = self.prev_wez_status[ag]
            curr_wez = self._in_wez(ag)
            self.prev_wez_status[ag] = curr_wez
            
            # WEZ entry bonus
            if not prev_wez and curr_wez:
                rewards[ag] += 0.1
            
            # Lock accumulation gated by WEZ and cooldown
            if curr_wez and (self.cooldown[ag] == 0):
                self.lock_counter[ag] += 1
                self.total_lock[ag] += 1
                rewards[ag] += 0.1  # Small bonus for maintaining lock
            else:
                self.lock_counter[ag] = 0

            # Shoot if lock requirement met
            if self.lock_counter[ag] >= self.lock_required and self.cooldown[ag] == 0:
                hits.append(ag)

        # Apply hits simultaneously
        for ag in hits:
            other = self._other(ag)
            rewards[ag] += 1.0  # Hit bonus
            rewards[other] -= 0.8  # Hit penalty
            self.hp[other] = max(0, int(self.hp[other]) - 10)
            self.lock_counter[ag] = 0
            self.cooldown[ag] = self.cooldown_frames
            self.episode_hits[ag] += 1

        # Cooldown tick
        for ag in self.agents:
            if self.cooldown[ag] > 0:
                self.cooldown[ag] -= 1

        # Termination conditions
        term = (
            self.current_step >= self.max_steps
            or self.hp['Agent 0'] <= 0
            or self.hp['Agent 1'] <= 0
        )
        
        # End of episode bonuses
        if term:
            for ag in self.agents:
                other = self._other(ag)
                # Survival bonus and enemy survival penalty
                if self.hp[ag] > 0 and self.hp[other] <= 0:
                    rewards[ag] += 5.0
                elif self.hp[ag] <= 0 and self.hp[other] > 0:
                    rewards[ag] -= 3.0
                    
                # Performance bonuses
                rewards[ag] += 0.1 * self.episode_hits[ag]  # Hit count bonus
                
        dones = {ag: term for ag in self.agents}
        dones['__all__'] = term

        obs = {ag: self._get_obs(ag) for ag in self.agents}
        global_state = np.concatenate([obs[a] for a in self.agents], axis=0)

        # Update previous distances
        for agent in self.agents:
            other = self._other(agent)
            self.prev_dist[agent] = float(np.linalg.norm(self.states[agent][:3] - self.states[other][:3]))

        # Infos
        for ag in self.agents:
            other = self._other(ag)
            infos[ag].update({
                'distance': float(np.linalg.norm(self.states[ag][:3] - self.states[other][:3])),
                'Total Lock': int(self.total_lock[ag]),
                'Agent 0 HP': int(self.hp['Agent 0']),
                'Agent 1 HP': int(self.hp['Agent 1']),
                'Episode Hits': int(self.episode_hits[ag]),
                'global_state': global_state.copy(),
            })

        return obs, rewards, dones, infos

    """ A helper to get the opposing agent's name """
    def _other(self, agent):
        return self.agents[1] if agent == self.agents[0] else self.agents[0]

    def _compute_reward(self, agent, weights=None):
        default_w = {
            # premi
            'point': 0.25,   # prua allineata alla LOS
            'tail':  0.05,   # il target è allineato alla LOS (noi dietro)
            'dist':  0.30,   # meglio corto raggio (scalato su WEZ)
            'close': 0.40,   # chiusura positiva

            # penalità
            'threat':      0.20,  # l'avversario ci sta puntando
            'threat_wez':  0.60,  # siamo dentro la sua WEZ

            # opzionali
            'wez_gate':  0.00,  # moltiplica la base quando NOI siamo in WEZ
            'wez_bonus': 0.00,  # bonus additivo quando NOI siamo in WEZ
        }
        w = default_w if (weights is None) else {**default_w, **weights}

        other = self._other(agent)

        los, d  = self._los(agent, other)     # i -> j
        ori_i   = self._orientation(agent)
        ori_j   = self._orientation(other)

        # Reward
        # Puntamento nostro -> LOS  (in [0,1])
        cos_point = float(np.clip(np.dot(ori_i['nose'], los), -1.0, 1.0))
        f_point   = 0.5 * (1.0 + cos_point)

        # Tail del target (se lui è allineato alla LOS, noi siamo dietro)
        cos_tail  = float(np.clip(np.dot(ori_j['nose'], los), -1.0, 1.0))
        f_tail    = 0.5 * (1.0 + cos_tail)

        # Distanza: corto raggio preferito, scala = WEZ
        wez_len   = float(self.wez_length)
        f_dist    = 1.0 / (1.0 + (d / max(1e-6, wez_len))**2)  # (0,1]

        # Closure
        c         = self._closure(agent, other)                # >0 se chiudo
        c_norm    = float(np.clip(c / max(1e-6, 2.0*self.v_max), -1.0, 1.0))
        f_close   = 0.5 * (1.0 + c_norm)

        # Penalties
        # Il nemico punta NOI (nose_j su -LOS)
        cos_threat = float(np.clip(np.dot(ori_j['nose'], -los), -1.0, 1.0))
        f_threat   = 0.5 * (1.0 + cos_threat)  # ~1 quando lui è allineato su di noi

        # Penalty if in enemy WEZ
        I_enemy_wez = 1.0 if self._in_wez(other) else 0.0

        # Agent WEZ
        I_my_wez = 1.0 if self._in_wez(agent) else 0.0
        
        # Coasting penalty, quando vanno in parallelo senza virare
        coast_gate = (cos_point > 0.85) and (f_close < 0.55) and (d > 0.5 * wez_len)
        if coast_gate:
            a = (cos_point - 0.85) / 0.15       # più allineato ⇒ peggio
            b = (0.55 - f_close) / 0.55         # meno agent chiude ⇒ peggio
            c = min(1.0, (d - 0.5*wez_len) / (0.5*wez_len))  # più lontano ⇒ peggio
            coast_pen = 0.15 * a * b * c
        else:
            coast_pen = 0.0

        base = (
            w['point'] * f_point +
            w['tail']  * f_tail  +
            w['dist']  * f_dist  +
            w['close'] * f_close
            - w['threat']     * f_threat
            - w['threat_wez'] * I_enemy_wez
        )

        reward = base * (1.0 + w['wez_gate'] * I_my_wez) + w['wez_bonus'] * I_my_wez
        reward -= coast_pen
        
        # Enemy aiming at us?
        cos_enemy_on_me = float(np.clip(np.dot(ori_j['nose'], -los), -1.0, 1.0))
        f_threat = 0.5 * (1.0 + cos_enemy_on_me)  # alto = grosso pericolo
        threat_k = 0.10
        threat_wez_k = 0.10
        reward -= threat_k * f_threat
        if self._in_wez(other):
            reward -= threat_wez_k * f_threat

        if hasattr(self, "prev_dist"):
            self.prev_dist[agent] = d

        return float(reward)


    def _nose_vec(self, agent):
        """Restituisce il vettore "nose" dell'agente."""
        _, _, _, _, yaw, pitch = self.states[agent]
        return np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ], dtype=np.float32)
        
    def _los(self, agent, target=None):
        """
        Line-Of-Sight (agent -> target).
        Ritorna (los_unit, distance).
        """
        if target is None:
            target = self._other(agent)

        p_i = self.states[agent][:3]
        p_j = self.states[target][:3]
        r = p_j - p_i
        d = float(np.linalg.norm(r))
        if d < 1e-8:
            # fallback arbitrario ma stabile
            return np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0
        return (r / d).astype(np.float32), d
    
    def _orientation(self, agent):
        """
        Orientazione e velocità dell'agent.
        Ritorna dict: {'yaw', 'pitch', 'nose', 'speed'}
        """
        _, _, _, v, yaw, pitch = self.states[agent]
        yaw   = float(yaw)
        pitch = float(pitch)
        nose = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch)
        ], dtype=np.float32)
        return {
            'yaw': yaw,
            'pitch': pitch,
            'nose': nose,
            'speed': float(v),
        }

    def _speed_vec(self, agent):
        """Vettore velocità v * nose."""
        o = self._orientation(agent)
        return o['speed'] * o['nose']

    def _closure(self, agent, target=None):
        """
        Closure rate lungo la LOS (positivo se la distanza diminuisce).
        """
        if target is None:
            target = self._other(agent)

        los, _ = self._los(agent, target)
        v_rel = self._speed_vec(agent) - self._speed_vec(target)
        return float(np.dot(v_rel, los))

    def _in_wez(self, shooter):
        """
        A function to check if an agent is within the Weapon Engagement Zone (WEZ).
        """
        target = self._other(shooter)
        rel = self.states[target][:3] - self.states[shooter][:3]
        d = np.linalg.norm(rel)
        if d < 1e-6:
            return False
        los = rel / d
        cos_nose = float(np.dot(self._nose_vec(shooter), los))
        return (d <= self.wez_length) and (cos_nose >= self.wez_cos_threshold)

    def render(self, mode='human'):
        print("Agent 0:", self.states['Agent 0'][:3], 
              "Agent 1:", self.states['Agent 1'][:3])