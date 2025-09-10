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
        # --- Arena bounds (hard walls) ---
        self.bound_half = np.array([45.0, 45.0, 50.0], dtype=np.float32)  # semiestensioni box su (x,y,z)
        self.bound_bounce_damping = 0.95   # smorzamento velocità dopo rimbalzo
        self.bound_eps = 1e-4              # piccolo margine per rientrare nel box
        
        # --- WEZ Parameters ---
        self.wez_length = 5.0
        self.wez_cos_threshold = float(np.cos(np.radians(30.0)))
        self.lock_required = 10 # For how many steps the agent needs to keep enemy in WEZ in order to shoot
        self.cooldown_frames = 5 # For how many steps the agent needs to wait before shooting again
        self.lock_counter = {ag: 0 for ag in self.agents}
        self.cooldown = {ag: 0 for ag in self.agents}
        self.coast_counter = {ag: 0 for ag in self.agents} # Counter used to avoid coasting
        # buffer eventi dell’ultimo step (usati da _compute_reward)
        self._last_hit_shooters = set()
        self._last_hit_victims  = set()
        # reward/event weights (più danno, meno farm)
        self.r_hit_reward  = 10.0   # premio se io colpisco
        self.r_hit_penalty = 6.0   # penalità se vengo colpito
        self.r_lock_tick   = 0.01  # shaping minimo per lock “buono”
        self.lead_tau_max  = 1.8   # orizzonte massimo per il lead
        
        
        self.pmd_lock_frac_start = 0.60   # all’inizio: PMD ammessa = 60% WEZ
        self.pmd_lock_frac_end   = 0.35   # a regime: 35% WEZ
        self.lead_lock_deg_start = 28.0   # più permissivo all’inizio
        self.lead_lock_deg_end   = 22.0   # più stretto a regime
        self.lock_required_start = 8
        self.lock_required_end   = 12
        self.r_lock_tick         = 0.005  # piccolo shaping SOLO su lock buono

        # flag step->reward per sapere se il frame era “good_geom”
        self._good_geom_flag = {ag: False for ag in self.agents}
        
        # --- Damping rimbalzi muro (fine-tuning pareti fisiche) ---
        if hasattr(self, "bound_bounce_damping"):
            self.bound_bounce_damping = 0.90  # prima 0.95


        # --- Agent controls ---
        self.delta_v = 0.15
        self.yaw_rate_max = 1.8             # [rad/s] yaw rate limit
        self.pitch_rate_max = 1.8           # [rad/s] pitch rate limit
        self.pitch_abs_max = math.radians(60.0)  # pitch limit
        self.v_max = 1.5
        self.v_min = 1.0
        # --- Riferimento per normalizzare ω_LOS ---
        self._losrate_ref = float(self.v_max / max(1e-6, self.wez_length))  # [rad/s] ~ v/WEZ

        # Observation space
        obs_dim = 21
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
        self.coast_counter = {ag: 0 for ag in self.agents}
        self._good_geom_flag = {ag: False for ag in self.agents}
        self._last_hit_shooters = set()
        self._last_hit_victims = set()
        self._diag = dict(los_rate_sum=0.0, pmd_sum=0.0, steps=0, good_lock_steps=0, lock_steps=0)

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

            pos = np.clip(pos, -self.bound_half + self.bound_eps, self.bound_half - self.bound_eps)
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
        - LOS unitario verso il target (3)
        - distanza normalizzata (d / (4*WEZ)) in [0,1] tagliata a 1
        - cos_point = nose_i · los           in [-1,1]
        - cos_tail  = nose_j · los           in [-1,1]
        - v_close (closure lungo LOS)        >0 se si chiude
        - I_wez (0/1)
        - v_i (velocità propria)
        - yaw_i, pitch_i
        - v_j (velocità avversario)
        - enemy_hp (normalizzato su 100)
        - aspect_to_me = nose_j · (-los)     in [-1,1]
        - my_lock_progress  in [0,1]
        - enemy_lock_progress in [0,1]
        - 
        """
        other = self._other(agent)

        # Geometria di base
        los, d   = self._los(agent, other)           # i -> j
        ori_i    = self._orientation(agent)
        ori_j    = self._orientation(other)
        v_i      = ori_i['speed']
        v_j      = ori_j['speed']

        # Allineamenti (in [-1,1])
        cos_point      = float(np.clip(np.dot(ori_i['nose'], los), -1.0, 1.0))       # io verso il target
        cos_tail       = float(np.clip(np.dot(ori_j['nose'], los), -1.0, 1.0))       # j scappa lungo LOS
        cos_enemy_on_me= float(np.clip(np.dot(ori_j['nose'], -los), -1.0, 1.0))      # j mi punta

        # Lead pursuit (unit vector che anticipa il target)
        lead_dir = self._lead_unit(agent, other)     # già normalizzata
        cos_lead = float(np.clip(np.dot(ori_i['nose'], lead_dir), -1.0, 1.0))

        # Closure normalizzata a [0,1]
        c       = self._closure(agent, other)
        c_norm  = float(np.clip(c / max(1e-6, 2.0*self.v_max), -1.0, 1.0))
        f_close = 0.5 * (1.0 + c_norm)

        # WEZ
        I_my_wez    = 1.0 if self._in_wez(agent) else 0.0
        I_enemy_wez = 1.0 if self._in_wez(other) else 0.0

        # Distanza normalizzata rispetto alla WEZ (clamp a 1)
        d_norm = float(min(d / max(1e-6, self.wez_length), 1.0))

        # Rapporto velocità (clamp a [0,1.5] e poi riscalo a [0,1])
        v_ratio = float(np.clip(v_i / max(0.1, v_j), 0.0, 1.5)) / 1.5
        
        los_rate, pmd = self._los_rate_and_pmd(agent, self._other(agent))
        los_rate_norm = float(np.clip(los_rate / max(1e-6, self._losrate_ref), 0.0, 1.0))
        pmd_norm = float(np.clip(pmd / max(1e-6, self.wez_length), 0.0, 1.0))
        
        # Componenti radiali, tangenziali della velocità e coplanarità dell'orbita
        vr, vt, cos_orb, d_now = self._relative_components(agent, self._other(agent))
        vr_norm = float(np.clip(vr / max(1e-6, self.v_max), -1.0, 1.0))          # [-1,1]
        vt_norm = float(np.clip(vt / max(1e-6, self.v_max), 0.0, 1.0))           # [0,1]
        cos_orb_norm = float(cos_orb)                                            # [-1,1]
        # opzionale: distanza alla “ring-radius” normalizzata (aiuta la stabilità)
        ring_r = 0.6 * float(self.wez_length)
        ring_err = float(np.clip((d_now - ring_r) / max(1e-6, ring_r), -1.0, 1.0))

        obs = np.array([
            # LOS (3)
            los[0], los[1], los[2],
            # distanza e velocità (2)
            d_norm, v_ratio,
            # allineamenti e lead (4)
            cos_point, cos_tail, cos_lead, cos_enemy_on_me,
            # closure (1)
            f_close,
            # WEZ (2)
            I_my_wez, I_enemy_wez,
            # velocità normalizzate assolute (2) per stabilità del controllo
            v_i / max(1e-6, self.v_max),
            v_j / max(1e-6, self.v_max),
            # margine laterale (1): componente perpendicolare della prua rispetto alla LOS
            float(np.linalg.norm(np.cross(ori_i['nose'], los))),
            los_rate_norm,  # tasso di variazione della LOS normalizzato (1)
            pmd_norm,       # miss distance predetta normalizzata (1)
            vr_norm,      # componente radiale normalizzata (1)
            vt_norm,      # componente tangenziale normalizzata (1)
            cos_orb_norm, # coplanarità dell'orbita (1)
            ring_err,     # errore sulla ring-radius normalizzato (1)
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

            # new = np.array([x + dx, y + dy, z + dz, v, yaw, pitch], dtype=np.float32)
            # self.states[ag] = new
            new = np.array([x + dx, y + dy, z + dz, v, yaw, pitch], dtype=np.float32)
            new = self._apply_bounds(new)
            self.states[ag] = new

        # Advance time
        self.current_step += 1

        # Shooting
        hits = []

        for ag in self.agents:
            prev_wez = self.prev_wez_status[ag]
            curr_wez = self._in_wez(ag)
            self.prev_wez_status[ag] = curr_wez
            
            # Lock accumulation gated by WEZ and cooldown
            good_geom = self._update_lock_gated(ag, curr_wez)
            self._good_geom_flag[ag] = bool(good_geom)
            if good_geom:
                self.total_lock[ag] += 1

            # Shoot if lock requirement met
            if self.lock_counter[ag] >= self.lock_required and self.cooldown[ag] == 0:
                hits.append(ag)

        # Apply hits simultaneously
        self._last_hit_shooters = set()
        self._last_hit_victims = set()
        for ag in hits:
            other = self._other(ag)
            self.hp[other] = max(0, int(self.hp[other]) - 10)
            self.lock_counter[ag] = 0
            self.cooldown[ag] = self.cooldown_frames
            self.episode_hits[ag] += 1
            self._last_hit_shooters.add(ag)
            self._last_hit_victims.add(other)

        # Cooldown tick
        for ag in self.agents:
            if self.cooldown[ag] > 0:
                self.cooldown[ag] -= 1
                
        rewards = {ag: self._compute_reward(ag) for ag in self.agents}
        self._last_hit_shooters.clear()
        self._last_hit_victims.clear()

        # Termination conditions
        term = (
            self.current_step >= self.max_steps
            or self.hp['Agent 0'] <= 0
            or self.hp['Agent 1'] <= 0
        )
                
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

    def _compute_reward(self, agent: str, weights: dict | None = None) -> float:
        """
        Reward pulita e non farmabile:
        - pesi spostati sugli HIT
        - una sola penalità 'threat'
        - anti-coast coerente e più leggero
        - shaping WEZ minimo + comfort ridotto
        - gestione degli eventi di tiro qui (hit a favore/contro)
        """
        other = self._other(agent)

        # --- base geometry ---
        los, d  = self._los(agent, other)
        ori_i   = self._orientation(agent)
        ori_j   = self._orientation(other)
        wez     = float(self.wez_length)
        v_max   = float(self.v_max)

        # allineamenti
        cos_point = float(np.clip(np.dot(ori_i['nose'], los), -1.0, 1.0))
        f_point   = 0.5 * (1.0 + cos_point)

        cos_tail  = float(np.clip(np.dot(ori_j['nose'], los), -1.0, 1.0))
        f_tail    = 0.5 * (1.0 + cos_tail)

        # lead pursuit (intercetto)
        lead_dir  = self._lead_unit(agent, other)
        cos_lead  = float(np.clip(np.dot(ori_i['nose'], lead_dir), -1.0, 1.0))
        f_lead    = 0.5 * (1.0 + cos_lead)

        # closure in [0,1]
        closure   = self._closure(agent, other)
        c_norm    = float(np.clip(closure / max(1e-6, 2.0*v_max), -1.0, 1.0))
        f_close   = 0.5 * (1.0 + c_norm)

        # banda "comfort" molto debole (picco poco fuori dalla WEZ)
        center    = 0.8 * wez
        width     = 0.5 * wez
        f_band    = float(np.exp(-0.5 * ((d - center) / max(1e-6, width))**2))  # 0..1

        # minaccia (singola)
        cos_threat  = float(np.clip(np.dot(ori_j['nose'], -los), -1.0, 1.0))
        f_threat    = 0.5 * (1.0 + cos_threat)
        I_enemy_wez = 1.0 if self._in_wez(other) else 0.0
        I_my_wez    = 1.0 if self._in_wez(agent) else 0.0

        # ---- somma pesata (ridotti i termini "farmabili") ----
        reward = (
            0.25 * f_point +
            0.03 * f_tail  +
            0.30 * f_close +
            0.22 * f_lead  +
            0.03 * f_band
        )
        # penalità
        reward -= 0.25 * f_threat
        reward -= 0.45 * I_enemy_wez

        # piccolo boost solo se davvero ingaggio “sensato” nella mia WEZ
        if I_my_wez and (closure > 0.0) and (cos_point >= self.wez_cos_threshold):
            reward += 0.02

        # shaping minimo per lock “buono” (non farmabile perché già gated sopra)
        if self._good_geom_flag.get(agent, False):
            reward += self.r_lock_tick

        # anti-coast più leggero (usa il contatore interno)
        reward -= 0.12 * self._anti_coasting(agent, los, d, ori_i, closure)
        
        los_u, d_ij = self._los(agent, self._other(agent))
        ori = self._orientation(agent)
        in_wez = (d_ij <= float(self.wez_length)) and (float(np.dot(ori['nose'], los_u)) >= float(self.wez_cos_threshold))
        if in_wez:
            los_rate, pmd = self._los_rate_and_pmd(agent, self._other(agent))
            f_pmd = math.exp(-0.5 * (pmd / max(1e-6, 0.5 * float(self.wez_length)))**2)
            f_los = float(np.clip(los_rate / max(1e-6, float(self._losrate_ref)), 0.0, 1.0))
            reward += 0.10 * f_pmd + 0.05 * f_los

        # ---- eventi di tiro (settati in step) ----
        if agent in self._last_hit_shooters:
            reward += self.r_hit_reward
        if agent in self._last_hit_victims:
            reward -= self.r_hit_penalty
            
        # ----- ORBITAL PURSUIT SHAPING -----
        vr, vt, cos_orb, d_now = self._relative_components(agent, other)
        I_my_wez = 1.0 if self._in_wez(agent) else 0.0

        # target anello
        r0      = 0.6 * float(self.wez_length)
        sig_r   = 0.18 * r0                # spessore “buono”
        ring_ok = math.exp(-0.5 * ((d_now - r0) / max(1e-6, sig_r))**2)  # 0..1

        # normalizzazioni robuste
        vr_n = np.clip(vr / max(1e-6, self.v_max), -1.0, 1.0)      # [-1,1]
        vt_n = np.clip(vt / max(1e-6, self.v_max), 0.0, 1.0)       # [0,1]
        cos_o = np.clip(cos_orb, -1.0, 1.0)                        # [-1,1]

        # desidero un “closing” lieve in anello (evita orbiting sterile)
        vr_t   = 0.10 * self.v_max
        sig_vr = 0.15 * self.v_max
        vr_ok  = math.exp(-0.5 * ((vr - vr_t) / max(1e-6, sig_vr))**2)     # 0..1

        if I_my_wez:
            # dentro WEZ: spingi a orbitare stretto e co-pianare, con lieve closing
            reward += 0.12 * ring_ok
            reward += 0.10 * float(vt_n)
            reward += 0.07 * float(0.5 * (1.0 + cos_o))  # mappa [-1,1] -> [0,1]
            reward += 0.08 * vr_ok
        else:
            # fuori WEZ: semplicità → chiudi e curva (porta dentro)
            reward += 0.05 * max(0.0, vr_n)     # solo closing
            reward += 0.03 * float(vt_n)

        # PMD / ω_LOS (già presenti): tienili, ma riduci a 0.08 / 0.04 se ora “spingono troppo”
        if I_my_wez:
            los_rate, pmd = self._los_rate_and_pmd(agent, self._other(agent))
            f_pmd = math.exp(-0.5 * (pmd / max(1e-6, 0.5 * float(self.wez_length)))**2)
            f_los = float(np.clip(los_rate / max(1e-6, float(self._losrate_ref)), 0.0, 1.0))
            reward += 0.08 * f_pmd + 0.04 * f_los

        # traccia distanza
        if hasattr(self, "prev_dist"):
            self.prev_dist[agent] = d

        return float(reward)
    
    def _apply_bounds(self, state: np.ndarray) -> np.ndarray:
        """
        Applica limiti fisici di un box axis-aligned:
        - Clippa la posizione al bordo
        - Riflette la direzione di moto rispetto alla normale del muro
        - Applica uno smorzamento di velocità (bound_bounce_damping)
        Restituisce lo stato modificato [x,y,z,v,yaw,pitch].
        """
        x, y, z, v, yaw, pitch = map(float, state)
        p = np.array([x, y, z], dtype=np.float64)
        H = self.bound_half.astype(np.float64)

        # velocità vettoriale corrente
        vx = v * math.cos(pitch) * math.cos(yaw)
        vy = v * math.cos(pitch) * math.sin(yaw)
        vz = v * math.sin(pitch)
        vv = np.array([vx, vy, vz], dtype=np.float64)

        collided = False

        # per ogni asse, clamp + riflessione
        for k in (0, 1, 2):
            if p[k] < -H[k]:
                p[k] = -H[k] + self.bound_eps
                n = np.zeros(3, dtype=np.float64); n[k] = 1.0   # normale verso l'interno
                vv = vv - 2.0 * np.dot(vv, n) * n
                collided = True
            elif p[k] > H[k]:
                p[k] = H[k] - self.bound_eps
                n = np.zeros(3, dtype=np.float64); n[k] = -1.0  # normale verso l'interno
                vv = vv - 2.0 * np.dot(vv, n) * n
                collided = True

        if collided:
            v_new = float(np.linalg.norm(vv))
            if v_new < 1e-8:
                # fallback stabile
                dir_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                v_new = float(self.v_min)
            else:
                dir_hat = vv / v_new
                v_new = float(np.clip(v_new * self.bound_bounce_damping, self.v_min, self.v_max))

            yaw = float(math.atan2(dir_hat[1], dir_hat[0]))
            pitch = float(math.atan2(dir_hat[2], math.sqrt(dir_hat[0]**2 + dir_hat[1]**2)))
            v = v_new

        return np.array([p[0], p[1], p[2], v, yaw, pitch], dtype=np.float32)

    def _lead_los(self, agent, target=None, k: float = 0.7, tau_max: float = 3.0):
        """
        Direzione di intercetto 'lead' (unitaria) dall'agent verso
        la posizione predetta del target.
        k: fattore per smorzare il tempo di anticipo.
        tau_max: bound sul tempo di anticipo.
        """
        if target is None:
            target = self._other(agent)

        # geometria attuale
        p_i = self.states[agent][:3]
        p_j = self.states[target][:3]
        (los, d) = self._los(agent, target)
        oi = self._orientation(agent)
        oj = self._orientation(target)

        # tempo di anticipo ~ distanza / nostra velocità (smorzato e clampato)
        v_i = max(1e-6, oi['speed'])
        tau = min(k * (d / v_i), tau_max)

        # posizione predetta del target e nuova LOS
        p_j_pred = p_j + oj['nose'] * oj['speed'] * tau
        r = p_j_pred - p_i
        n = float(np.linalg.norm(r))
        if n < 1e-8:
            return los  # fallback stabile
        return (r / n).astype(np.float32)

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
        return -float(np.dot(v_rel, los))

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

    def set_curriculum(self, progress: float):
        """
        progress in [0,1]: da facile a difficile.
        - Distanze di start più grandi e orientamenti meno favorevoli col progredire.
        - WEZ si stringe leggermente e il lock richiede più passi.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        # parametri cinematici
        self.yaw_rate_max   = 1.2 + 0.6 * p       # più agile con il tempo
        self.pitch_rate_max = 1.0 + 0.5 * p
        # WEZ/lock
        self.wez_length     = 6.0 - 1.5 * p       # WEZ un po' più stretto col tempo
        self.lock_required  = int(6 + 6 * p)      # da 6 a 12
        # memorizza per reset
        self._curriculum_p  = p

    def _anti_coasting(self, agent, los, d, ori_i, closure):
        """
        Penalità progressiva se l'agente 'veleggia' allineato ma senza chiudere.
        Ritorna un valore in [0,1] da sottrarre alla reward (scale esterna).
        """
        wez = float(self.wez_length)
        cos_point = float(np.clip(np.dot(ori_i['nose'], los), -1.0, 1.0))
        # gating: molto allineato, poca chiusura, abbastanza lontano
        is_coast = (cos_point > 0.9) and (closure < 0.02) and (d > 0.75 * wez)
        if is_coast:
            self.coast_counter[agent] = min(self.coast_counter[agent] + 1, 50)
        else:
            self.coast_counter[agent] = max(self.coast_counter[agent] - 2, 0)
        x = self.coast_counter[agent] / 50.0
        return x * x  # curva dolce 0..1

    def _lead_unit(self, agent: str, other: str, t_lead: float | None = None) -> np.ndarray:
        """
        Direzione di ingaggio 'lead' (unitaria) dal punto di vista di `agent`,
        puntando alla posizione futura stimata del bersaglio `other`.

        t_lead default: tempo necessario a coprire la distanza attuale con v_max.
        """
        s_i = self.states[agent]
        s_j = self.states[other]

        p_i = s_i[:3].astype(np.float64)
        p_j = s_j[:3].astype(np.float64)

        # yaw/pitch -> versore velocità (assumiamo modulo ~ v_max)
        yaw_j   = float(s_j[4]); pitch_j = float(s_j[5])
        cp = np.cos(pitch_j)
        vhat_j = np.array([cp*np.cos(yaw_j), cp*np.sin(yaw_j), np.sin(pitch_j)], dtype=np.float64)

        rel = p_j - p_i
        d = float(np.linalg.norm(rel))
        if d < 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        if t_lead is None:
            t_lead = d / max(1e-6, float(self.v_max))  # grezza ma stabile

        p_j_future = p_j + vhat_j * float(self.v_max) * t_lead
        v = p_j_future - p_i
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else rel / d

    def _los_rate_and_pmd(self, agent: str, target: str | None = None) -> tuple[float, float]:
        if target is None:
            target = self._other(agent)
        p_i = self.states[agent][:3].astype(np.float64)
        p_j = self.states[target][:3].astype(np.float64)
        v_i = self._speed_vec(agent)
        v_j = self._speed_vec(target)
        r = p_j - p_i
        d2 = float(np.dot(r, r)) + 1e-12
        v_rel = v_j - v_i
        # |ω_LOS| = || v_rel × r || / |r|^2
        los_rate = float(np.linalg.norm(np.cross(v_rel, r)) / d2)
        # PMD = || r × v_rel || / || v_rel ||
        vrel_n = float(np.linalg.norm(v_rel)) + 1e-12
        pmd = float(np.linalg.norm(np.cross(r, v_rel)) / vrel_n)
        return los_rate, pmd

    def _update_lock_gated(self, agent: str, curr_wez: bool) -> bool:
        """Aggiorna lock_counter con gating di qualità. Ritorna good_geom (bool)."""
        tgt = self._other(agent)
        los_u, _ = self._los(agent, tgt)
        ori = self._orientation(agent)  # deve avere 'nose'
        lead_u = self._lead_unit(agent, tgt)
        closure = self._closure(agent, tgt)
        _, pmd = self._los_rate_and_pmd(agent, tgt)
        
        pmd_thr  = float(getattr(self, "pmd_lock_frac", self.pmd_lock_frac_end)) * float(self.wez_length)
        lead_cos = float(np.cos(np.radians(getattr(self, "lead_lock_deg", self.lead_lock_deg_end))))

        good_geom = (
            bool(curr_wez) and
            (self.cooldown[agent] == 0) and
            (closure > 0.0) and
            (float(np.dot(ori['nose'], los_u))  >= float(self.wez_cos_threshold)) and
            (float(np.dot(ori['nose'], lead_u)) >= lead_cos) and
            (pmd <= pmd_thr)
        )
        if good_geom:
            self.lock_counter[agent] += 1
        else:
            self.lock_counter[agent] = 0
        return good_geom

    def _relative_components(self, agent: str, target: str | None = None):
        if target is None:
            target = self._other(agent)
        p_i = self.states[agent][:3].astype(np.float64)
        p_j = self.states[target][:3].astype(np.float64)
        v_i = self._speed_vec(agent).astype(np.float64)
        v_j = self._speed_vec(target).astype(np.float64)
        r = p_j - p_i
        d = float(np.linalg.norm(r)) + 1e-12
        los = r / d
        v_rel = v_i - v_j
        vr = -float(np.dot(v_rel, los))                         # >0 se chiudi
        vt = float(np.linalg.norm(np.cross(v_rel, los)))        # tangenziale
        h_i = np.cross(r, v_i); h_j = np.cross(r, v_j)
        nh_i = np.linalg.norm(h_i); nh_j = np.linalg.norm(h_j)
        cos_orbit = 0.0
        if nh_i > 1e-9 and nh_j > 1e-9:
            cos_orbit = float(np.clip(np.dot(h_i, h_j) / (nh_i * nh_j), -1.0, 1.0))
        return vr, vt, cos_orbit, d

    def render(self, mode='human'):
        print("Agent 0:", self.states['Agent 0'][:3], 
              "Agent 1:", self.states['Agent 1'][:3])