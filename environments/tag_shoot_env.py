
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

        # >>> Disimpegno & soft walls (1d)
        self.disengage_dist_factor = 3.0
        self.disengage_steps_required = 50
        self.disengage_counter = {ag: 0 for ag in self.agents}
        self.soft_boundary = 30.0  # raggio "morbido" per x/y/z
        
        # >>> Eventi colpo (1b)
        self.hit_reward_shooter = 6.0
        self.hit_penalty_target = 7.0

        # >>> Stato per 1b/1c/1e
        self.prev_lock_progress = {ag: 0.0 for ag in self.agents}         # my lock (step precedente)
        self.prev_enemy_lock_progress = {ag: 0.0 for ag in self.agents}   # enemy lock (step precedente)
        self.prev_yaw = {ag: 0.0 for ag in self.agents}                   # per stimare yaw_rate
        self.hit_decay = 0.9                                              # 1e
        self.decayed_hit = {ag: 0.0 for ag in self.agents}                # 1e
        self.prev_progress = {ag: 0.0 for ag in self.agents}

        # Observation space
        obs_dim = 20
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
        self.disengage_counter = {ag: 0 for ag in self.agents}
        self.decayed_hit = {ag: 0.0 for ag in self.agents}
        self.prev_lock_progress = {ag: 0.0 for ag in self.agents}
        self.prev_enemy_lock_progress = {ag: 0.0 for ag in self.agents}
        self.prev_progress = {ag: 0.0 for ag in self.agents}

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
        self.prev_yaw = {ag: 0.0 for ag in self.agents}  # yaw iniziale
        
        # Assign initial state to each agent
        for agent in self.agents:
            if agent == 'Agent 0':
                pos, yaw, pitch = ch_pos, yaw_cp, pitch_cp
            else:
                pos, yaw, pitch = ep_pos, yaw_ep, pitch_ep

            state = np.array([*pos, speed, yaw, pitch], dtype=np.float32)
            self.states[agent] = state
            self.prev_yaw[agent] = float(yaw)

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
        enemy_lock_progress = float(self.lock_counter[other] / max(1, self.lock_required))

        # Distanza normalizzata su scala 4*WEZ
        d_norm = float(min(d / max(1e-6, 4.0 * self.wez_length), 1.0))

        # --- 1e: nuove feature osservabili ---
        enemy_cooldown_norm = float(self.cooldown[other] / max(1, self.cooldown_frames))
        was_hit_decay = float(self.decayed_hit[agent])  # [0,1] con decadimento esponenziale
        d_my_lock = float(np.clip(my_lock_progress - self.prev_lock_progress[agent], -1.0, 1.0))
        d_enemy_lock = float(np.clip(enemy_lock_progress - self.prev_enemy_lock_progress[agent], -1.0, 1.0))

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
            enemy_lock_progress,              # 15
            enemy_cooldown_norm,              # 16  (NUOVO)
            was_hit_decay,                    # 17  (NUOVO)
            d_my_lock,                        # 18  (NUOVO)
            d_enemy_lock                      # 19  (NUOVO)
        ], dtype=np.float32)

        return obs

    def step(self, action_dict):
        rewards = {ag: 0.0 for ag in self.agents}
        infos = {ag: {} for ag in self.agents}

        # Update agents states (come prima: smoothing + integrazione)
        for ag, act in action_dict.items():
            st = self.states[ag].copy()

            # Smooth incremental action
            prev = self.prev_act[ag]
            smoothed = self.alpha_act * prev + (1.0 - self.alpha_act) * np.asarray(act, dtype=np.float32)
            self.prev_act[ag] = smoothed.copy()

            x, y, z, v, yaw, pitch = st
            dv, yaw_rate, pitch_rate = smoothed

            v_min, v_max = float(self.v_min), float(self.v_max)
            v = float(np.clip(v + dv, v_min, v_max))

            yaw = float(((yaw + float(np.clip(yaw_rate, -self.yaw_rate_max, self.yaw_rate_max)) * self.dt + np.pi) % (2.0 * np.pi)) - np.pi)
            pitch = float(np.clip(
                pitch + float(np.clip(pitch_rate, -self.pitch_rate_max, self.pitch_rate_max)) * self.dt,
                -self.pitch_abs_max, self.pitch_abs_max
            ))

            dx = v * math.cos(pitch) * math.cos(yaw) * self.dt
            dy = v * math.cos(pitch) * math.sin(yaw) * self.dt
            dz = v * math.sin(pitch) * self.dt

            new = np.array([x + dx, y + dy, z + dz, v, yaw, pitch], dtype=np.float32)
            self.states[ag] = new

        # Advance time
        self.current_step += 1

        # Disimpegno: conta quando la distanza supera soglia (1d)
        # Distanze correnti
        dist_now = {}
        for ag in self.agents:
            other = self._other(ag)
            d = float(np.linalg.norm(self.states[ag][:3] - self.states[other][:3]))
            dist_now[ag] = d

        # Reward base
        for ag in self.agents:
            rewards[ag] = float(self._compute_reward(ag))
            # Penalità lieve se troppo lontani (> 3*WEZ), per step
            if dist_now[ag] > 3.0 * self.wez_length:
                overflow = (dist_now[ag] - 3.0 * self.wez_length) / max(1e-6, self.wez_length)
                rewards[ag] -= 0.02 * min(3.0, overflow)   # ~0.02–0.06 per step

        # Shooting (1b): lock gain/rafforzato + eventi colpo con flag "hit" per 1e
        hits = []
        for ag in self.agents:
            prev_wez = self.prev_wez_status[ag]
            curr_wez = self._in_wez(ag)
            self.prev_wez_status[ag] = curr_wez

            # WEZ entry bonus (lieve)
            if not prev_wez and curr_wez:
                rewards[ag] += 0.1

            # Lock accumulation gated by WEZ and cooldown
            if curr_wez and (self.cooldown[ag] == 0):
                self.lock_counter[ag] += 1
                self.total_lock[ag] += 1
                # piccolo shaping additivo per mantenere il lock
                rewards[ag] += 0.1
            else:
                self.lock_counter[ag] = 0

            # Shoot se lock requirement soddisfatto
            if self.lock_counter[ag] >= self.lock_required and self.cooldown[ag] == 0:
                hits.append(ag)

        # Applica colpi simultaneamente (1b) + aggiorna segnale "was_hit" (1e)
        for ag in hits:
            other = self._other(ag)
            rewards[ag] += self.hit_reward_shooter     # +6
            rewards[other] -= self.hit_penalty_target  # -7
            self.hp[other] = max(0, int(self.hp[other]) - 10)
            self.lock_counter[ag] = 0
            self.cooldown[ag] = self.cooldown_frames
            self.episode_hits[ag] += 1
            self.decayed_hit[other] = 1.0  # impulsa il segnale "sono stato colpito"

        # Cooldown tick
        for ag in self.agents:
            if self.cooldown[ag] > 0:
                self.cooldown[ag] -= 1

        # Termination conditions
        term_time = (self.current_step >= self.max_steps)
        term_hp = (self.hp['Agent 0'] <= 0) or (self.hp['Agent 1'] <= 0)
        term = term_time or term_hp

        # End of episode shaping (1b/1d)
        if term:
            # Differenziale HP (premia chi ha più HP residua)
            for ag in self.agents:
                other = self._other(ag)
                rewards[ag] += 0.1 * float(self.hp[ag] - self.hp[other])  # 1b
                rewards[ag] += 0.1 * self.episode_hits[ag]                # tieni il bonus hit-count

        dones = {ag: term for ag in self.agents}
        dones['__all__'] = term

        # Osservazioni aggiornate
        obs = {ag: self._get_obs(ag) for ag in self.agents}
        global_state = np.concatenate([obs[a] for a in self.agents], axis=0)

        # Aggiorna "previous" per la prossima step (1b/1c/1e)
        for ag in self.agents:
            other = self._other(ag)
            self.prev_dist[ag] = float(np.linalg.norm(self.states[ag][:3] - self.states[other][:3]))
            self.prev_yaw[ag] = float(self.states[ag][4])
            self.prev_lock_progress[ag] = float(self.lock_counter[ag] / max(1, self.lock_required))
            self.prev_enemy_lock_progress[ag] = float(self.lock_counter[other] / max(1, self.lock_required))
            self.decayed_hit[ag] *= self.hit_decay  # decadimento (1e)

        # Infos
        for ag in self.agents:
            other = self._other(ag)
            infos[ag].update({
                'distance': dist_now[ag],
                'Total Lock': int(self.total_lock[ag]),
                'Agent 0 HP': int(self.hp['Agent 0']),
                'Agent 1 HP': int(self.hp['Agent 1']),
                'Episode Hits': int(self.episode_hits[ag]),
                'global_state': global_state.copy(),
            })

        return obs, rewards, dones, infos

    def _compute_reward(self, agent, *_):
        """
        Reward "a fasi" per evitare il volo parallelo e guidare l'ingaggio:
        - FAR   (d > 2*WEZ): priorità CHIUSURA + puntare la LOS (point)
        - MID   (WEZ < d ≤ 2*WEZ): costruzione della coda (point + behind), front_pen moderata
        - CLOSE (d ≤ WEZ): stare dietro e mantenere la posizione (behind forte, front_pen forte), min. z-gap,
                            minaccia attiva solo se l'altro ci punta davvero in WEZ
        - Anti-parallel: penalizza "stesso verso" + scarso point + poca chiusura a distanza medio/lunga
        - Lock shaping minimo: piccolo bonus step in WEZ ben puntati + bonus al differenziale di lock
        """
        other = self._other(agent)

        # --- Geometria di base ---
        agent_pos  = self.states[agent][:3]
        target_pos = self.states[other][:3]
        los, d     = self._los(agent, other)             # i -> j
        ori_i      = self._orientation(agent)
        ori_j      = self._orientation(other)

        my_dir     = ori_i['nose']
        target_dir = ori_j['nose']

        # vettore target->agente e "dietro/avanti" rispetto alla direzione del target
        vec = agent_pos - target_pos
        d_safe = float(max(np.linalg.norm(vec), 1e-8))
        pos_dir = vec / d_safe
        cos_pos = float(np.clip(np.dot(pos_dir, target_dir), -1.0, 1.0))  # >0 = siamo davanti
        behind  = 0.5 * (1.0 - cos_pos)                                   # grande se dietro [0,1]
        front_pen = max(0.0, cos_pos)                                      # penalità di front

        # puntare la LOS (ruotare verso il bersaglio)
        cos_point = float(np.clip(np.dot(my_dir, los), -1.0, 1.0))
        point = 0.5 * (1.0 + cos_point)                                    # [0,1]

        # stessa rotta del target (serve per anti-parallel)
        cos_vel = float(np.clip(np.dot(my_dir, target_dir), -1.0, 1.0))

        # chiusura positiva
        c = self._closure(agent, other)
        c_norm = float(np.clip(c / max(1e-6, 2.0*self.v_max), -1.0, 1.0))
        close_pos = max(0.0, c_norm)                                       # solo se stiamo chiudendo

        # z-gap (riduce fuga in quota quando vicini)
        z_gap_norm = abs(float(self.states[agent][2] - self.states[other][2])) / max(1e-6, self.wez_length)
        z_gap_norm = float(np.clip(z_gap_norm, 0.0, 1.5))

        # minaccia SOLO in WEZ nemica e se davvero ci punta
        I_enemy_wez = 1.0 if self._in_wez(other) else 0.0
        cos_enemy_on_me = float(np.clip(np.dot(target_dir, -los), -1.0, 1.0))
        threat_hard_gate = 1.0 if (I_enemy_wez == 1.0 and cos_enemy_on_me > 0.95) else 0.0
        f_threat = 0.5 * (1.0 + cos_enemy_on_me)                            # [0,1]

        # --- Anti-parallel (evita volo affiancato) ---
        # alto quando: stesse direzioni, scarso point, poca chiusura, distanza non troppo corta
        wez = float(self.wez_length)
        s_align = max(0.0, (cos_vel - 0.80) / 0.20)                         # 0..1 se molto allineati
        s_unpoint = max(0.0, (0.60 - point) / 0.60)                          # 0..1 se non stiamo puntando la LOS
        s_far = float(np.clip((d - 0.6*wez) / (0.8*wez), 0.0, 1.0))          # attivo da 0.6*WEZ a 1.4*WEZ+
        s_noclose = 1.0 - close_pos                                          # 1 se non stiamo chiudendo
        anti_parallel = s_align * s_unpoint * s_far * s_noclose              # 0..1

        # --- Fasi per distanza ---
        far_thr  = 2.0 * wez
        near_thr = 1.0 * wez

        reward = 0.0

        if d > far_thr:
            # LONTANO: chiudi e punta la LOS (niente front/threat)
            reward += 0.60 * close_pos + 0.40 * point
            reward -= 0.30 * anti_parallel

        elif d > near_thr:
            # MEDIO: costruisci coda, evita parallelo
            reward += 0.40 * point + 0.30 * behind + 0.30 * close_pos
            reward -= 0.30 * front_pen
            reward -= 0.35 * anti_parallel

        else:
            # VICINO: stare dietro e mantenere
            reward += 0.50 * behind + 0.30 * point + 0.10 * close_pos
            reward -= 0.60 * front_pen
            reward -= 0.10 * float(np.clip(z_gap_norm, 0.0, 1.0))
            reward -= 0.50 * threat_hard_gate * f_threat                     # minaccia forte solo se reale

            # Lock shaping minimo: piccolo bonus se in WEZ e ben puntati, + differenziale di lock
            I_my_wez = 1.0 if self._in_wez(agent) else 0.0
            if cos_point > math.cos(math.radians(20.0)) and I_my_wez == 1.0:
                reward += 0.02
            # bonus sul differenziale di lock progress
            if not hasattr(self, "_prev_lock_norm"):
                self._prev_lock_norm = {a: 0.0 for a in self.agents}
            my_lp = float(self.lock_counter[agent] / max(1, self.lock_required))
            d_my_lp = max(0.0, my_lp - float(self._prev_lock_norm.get(agent, 0.0)))
            reward += 0.05 * d_my_lp
            self._prev_lock_norm[agent] = my_lp

        return float(reward)

    def _other(self, agent):
        return self.agents[1] if agent == self.agents[0] else self.agents[0]

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

    def _angle_diff(self, a: float, b: float) -> float:
        # ritorna a-b in (-pi, pi]
        return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)

    def render(self, mode='human'):
        print("Agent 0:", self.states['Agent 0'][:3], 
              "Agent 1:", self.states['Agent 1'][:3])