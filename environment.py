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
    """
    def __init__(self, render_mode=None):
        super(PointMassEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.dt = 0.1
        self.max_steps = 1000
        self.current_step = 0
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.last_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Parametri per aggiornare lo stato
        self.delta_v = 0.25      # incremento velocità
        self.delta_theta = 0.25  # incremento angolo
        self.delta_z = 0.25      # incremento quota
        # self.v_max = 5.0       # velocità massima

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
        return self.full_state(), {}
    
    def full_state(self):
        return np.concatenate([self.state, self.target])

    def compute_reward(self, prev_state, current_state, target):
        # Estrai la posizione corrente (x, y, z)
        curr_pos = current_state[:3]
        
        # Calcola la distanza dal target
        curr_distance = np.linalg.norm(curr_pos - target)
        
        # Reward: penalizziamo ogni step in base alla distanza attuale,
        # quindi più sei lontano, più il reward è negativo.
        reward = -curr_distance
        
        # Bonus: se il target viene raggiunto (distanza inferiore a 0.5), aggiungi un bonus positivo
        if curr_distance < 0.5:
            reward += 100.0

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
        elif action == 4:   # Salire e Girare a destra
            z += self.delta_z
            theta += self.delta_theta
        elif action == 5:   # Salire e Girare a sinistra
            z += self.delta_z
            theta -= self.delta_theta
        elif action == 6:   # Scendere e Girare a destra
            z -= self.delta_z
            theta += self.delta_theta
        elif action == 7:   # Scendere e Girare a sinistra
            z -= self.delta_z
            theta -= self.delta_theta
        elif action == 8:   # Mantieni
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
        if current_distance < 0.5:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        self.last_distance = current_distance
        return self.full_state(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.render_mode == "human":
            print("Distanza dal target:", np.linalg.norm(self.state[:3] - self.target))