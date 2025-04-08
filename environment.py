import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PointMassEnv(gym.Env):
    """
    Ambiente semplificato per un punto materiale in 3D.
    Stato: [x, y, z, v, theta]
      - (x, y, z): posizione
      - v: velocità scalare
      - theta: orientamento nel piano xy (radianti)
    Azioni:
      0: Salire (incrementa z)
      1: Scendere (decrementa z)
      2: Girare a destra (incrementa theta)
      3: Girare a sinistra (decrementa theta)
      4: Accelerare (incrementa v)
      5: Frenare (diminuisce v)
    """
    def __init__(self, render_mode=None):
        super(PointMassEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.dt = 0.1
        self.max_steps = 1000
        self.current_step = 0
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.last_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Parametri per aggiornare lo stato
        self.delta_v = 0.25      # incremento velocità
        self.delta_theta = 0.25  # incremento angolo
        self.delta_z = 0.25      # incremento quota
        self.v_max = 5.0       # velocità massima

        # Target fisso
        self.target = np.array([10.0, 10.0, 5.0])
        self.last_distance = np.linalg.norm(self.state[:3] - self.target)

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reimposta l'ambiente e restituisce lo stato.
        
        Parametri:
            seed (int, optional): seed per il generatore di numeri casuali.
            options (dict, optional): opzioni aggiuntive.
        Ritorna:
            tuple: stato iniziale e informazioni aggiuntive.
        """
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.last_distance = np.linalg.norm(self.state[:3] - self.target)
        return self.state.copy(), {}

    def compute_reward(self, prev_state, current_state, target):
        # Estrai le posizioni 3D dagli stati
        
        prev_pos = prev_state[:3]
        curr_pos = current_state[:3]
 
        # Calcola le distanze dal target
        prev_distance = np.linalg.norm(prev_pos - target)
        curr_distance = np.linalg.norm(curr_pos - target)
 
        # Reward per il progresso: premio proporzionale alla riduzione della distanza
        improvement = prev_distance - curr_distance
        reward = max(improvement * 10.0, 0.0)  + 1/curr_distance

        return reward

    def step(self, action):
        """
        Aggiorna lo stato dell'ambiente eseguendo un passo di simulazione.
        
        Parametri:
            action (int): azione scelta dall'agente
        
        Ritorna:
            tuple: stato successivo, reward, terminated, truncated, info.
        """
        x, y, z, v, theta = self.state
        self.last_state = self.state.copy()

        if action == 0:      # Salire
            z += self.delta_z
        elif action == 1:    # Scendere
            z -= self.delta_z
        elif action == 2:    # Girare a destra
            theta += self.delta_theta
        elif action == 3:    # Girare a sinistra
            theta -= self.delta_theta
        elif action == 4:    # Accelerare
            v = min(v + self.delta_v, self.v_max)
        elif action == 5:    # Frenare
            v = max(v - self.delta_v, 0.0)
        elif action == 6:    # Salire e Accelerare
            z += self.delta_z
            v = min(v + self.delta_v, self.v_max)
        elif action == 7:    # Scendere e Frenare
            z -= self.delta_z
            v = max(v - self.delta_v, 0.0)
        elif action == 8:    # Girare a destra e Accelerare
            theta += self.delta_theta
            v = min(v + self.delta_v, self.v_max)
        elif action == 9:    # Girare a sinistra e Accelerare
            theta -= self.delta_theta
            v = min(v + self.delta_v, self.v_max)
        elif action == 10:   # Girare a destra e Frenare
            theta += self.delta_theta
            v = max(v - self.delta_v, 0.0)
        elif action == 11:   # Girare a sinistra e Frenare
            theta -= self.delta_theta
            v = max(v - self.delta_v, 0.0)
        elif action == 12:   # Salire e Girare a destra
            z += self.delta_z
            theta += self.delta_theta
        elif action == 13:   # Salire e Girare a sinistra
            z += self.delta_z
            theta -= self.delta_theta
        elif action == 14:   # Scendere e Girare a destra
            z -= self.delta_z
            theta += self.delta_theta
        elif action == 15:   # Scendere e Girare a sinistra
            z -= self.delta_z
            theta -= self.delta_theta
        elif action == 16:   # Mantieni
            pass
        

        # Aggiornamento posizione
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        self.state = np.array([x, y, z, v, theta], dtype=np.float32)
        self.current_step += 1
        current_distance = np.linalg.norm(self.state[:3] - self.target)

        reward = self.compute_reward(self.last_state, self.state, self.target)

        terminated = False
        truncated = False
        #if self.current_step > 200:
            #if current_distance > self.last_distance + 2.0:
                #truncated = True
                #print("EARLY STOP!!!")
        if current_distance < 0.5:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        self.last_distance = current_distance
        return self.state.copy(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.render_mode == "human":
            print("Distanza dal target:", np.linalg.norm(self.state[:3] - self.target))