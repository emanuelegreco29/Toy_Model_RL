import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.delta_theta = 0.15  # incremento angolo (radianti)
        self.delta_z = 0.25      # incremento quota
        self.v_max = 15       # velocità massima

        # Target fisso (da poter modificare in seguito)
        self.target = np.array([10.0, 10.0, 5.0])
        self.last_distance = np.linalg.norm(self.state[:3] - self.target)

        self.reset()

    def reset(self, seed=None, options=None):
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
        prev_theta = prev_state[4]

        # Calcola le distanze dal target
        prev_distance = np.linalg.norm(prev_pos - target)
        curr_distance = np.linalg.norm(curr_pos - target)

        reward = 0.0

        # 1) Ricompensa se la distanza è diminuita
        if curr_distance < prev_distance:
            # Differenza di distanza, moltiplicata per un fattore
            reward += (prev_distance - curr_distance) * 10.0
        else:
            # Penalità leggera (o 0) se non c’è miglioramento
            reward -= 1.0

        # Calcola il theta desiderato (direzione verso il target in XY)
        direction_to_target = target[:2] - prev_pos[:2]
        desired_theta = np.arctan2(direction_to_target[1], direction_to_target[0])
        current_theta = prev_state[4]
        theta_diff = np.abs((current_theta - desired_theta + np.pi) % (2 * np.pi) - np.pi)
        reward -= 0.5 * theta_diff  # penalizza grandi differenze di orientamento

        # 2) Calcola la variazione nel piano XY
        direction_xy = current_state[:2] - prev_state[:2]
        to_target_xy = target[:2] - prev_state[:2]
        norm_direction_xy = np.linalg.norm(direction_xy)
        norm_to_target_xy = np.linalg.norm(to_target_xy)
        if norm_direction_xy > 1e-6 and norm_to_target_xy > 1e-6:
            # Calcola il coseno dell'angolo tra il movimento e la direzione al target (proiezione xy)
            alignment_xy = np.dot(direction_xy, to_target_xy) / (norm_direction_xy * norm_to_target_xy)
            # Reward proporzionale allo spostamento in xy e all'allineamento
            reward += alignment_xy * norm_direction_xy * 5.0
            # Se l'allineamento è negativo, penalizza il movimento in direzione opposta
            if alignment_xy < 0:
                reward -= 5.0 * norm_direction_xy

        # 3) Bonus se vicinissimo
        if curr_distance < 2.0:
            reward += 5.0
        if curr_distance < 1.0:
            reward += 15.0

        # 4) Penalità se molto lontano
        if curr_distance > 50.0:
            reward -= 2.0

        reward = np.clip(reward, -100.0, 100.0)

        return reward

    def step(self, action):
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

        # Aggiornamento posizione
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        self.state = np.array([x, y, z, v, theta], dtype=np.float32)
        self.current_step += 1
        current_distance = np.linalg.norm(self.state[:3] - self.target)

        reward = self.compute_reward(self.last_state, self.state, self.target)

        terminated = False
        truncated = False
        if self.current_step > 200:
            if current_distance > self.last_distance + 2.0:  # peggiora
                truncated = True
        if current_distance < 0.5:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        self.last_distance = current_distance
        return self.state.copy(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.render_mode == "human":
            print("Distanza dal target:", np.linalg.norm(self.state[:3] - self.target))