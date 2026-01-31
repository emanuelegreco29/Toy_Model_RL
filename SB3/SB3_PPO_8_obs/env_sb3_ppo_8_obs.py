import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PointMassEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(PointMassEnv, self).__init__()
        self.dt = 0.1
        self.max_steps = 500
        self.current_step = 0

        # limiti per Δθ, Δz
        self.delta_theta = 0.5
        self.delta_z = 0.25

        # Action space continuo: [Δθ, Δz]
        low = np.array([-self.delta_theta, -self.delta_z], dtype=np.float32)
        high = np.array([self.delta_theta, self.delta_z], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # OBSERVATION SPACE 8-D: [x,y,z,v,θ, x_t,y_t,z_t]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # stato iniziale e target di default
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        self.target = np.array([10.0, 10.0, 5.0], dtype=np.float32)  # default (verrà sovrascritto a reset)

        self.target_max_init_distance = 25.0   # distanza massima iniziale dal punto di partenza
        self.target_min_init_distance = 2.0    # opzionale: evita target troppo vicini
        self.target_xy_bounds = (-30.0, 30.0)  # clamp opzionale per X,Y
        self.target_z_bounds = (0.0, 10.0)     # clamp opzionale per Z

        # per metriche episodio
        self._start_distance = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.state = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # genera un target casuale entro la distanza massima
        pos0 = self.state[:3].astype(np.float32)

        for _ in range(1000):
            v = np.random.normal(size=3).astype(np.float32)
            n = float(np.linalg.norm(v))
            if n < 1e-8:
                continue
            v /= n
            r = np.random.uniform(self.target_min_init_distance, self.target_max_init_distance)
            tgt = pos0 + v * float(r)

            tgt[0] = float(np.clip(tgt[0], self.target_xy_bounds[0], self.target_xy_bounds[1]))
            tgt[1] = float(np.clip(tgt[1], self.target_xy_bounds[0], self.target_xy_bounds[1]))
            tgt[2] = float(np.clip(tgt[2], self.target_z_bounds[0], self.target_z_bounds[1]))

            if float(np.linalg.norm(tgt - pos0)) <= self.target_max_init_distance + 1e-6:
                self.target = tgt.astype(np.float32)
                break

        self._start_distance = float(np.linalg.norm(self.state[:3] - self.target))
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.state, self.target]).astype(np.float32)

    @staticmethod
    def _distance(pos: np.ndarray, target: np.ndarray) -> float:
        return float(np.linalg.norm(pos - target))

    def compute_reward(self, prev_state: np.ndarray, current_state: np.ndarray, target: np.ndarray):
        curr_pos = current_state[:3]
        prev_pos = prev_state[:3]

        curr_dist = self._distance(curr_pos, target)
        prev_dist = self._distance(prev_pos, target)

        improvement = prev_dist - curr_dist

        # component 1: shaping su improvement
        if improvement > 0:
            improvement_term = 1.0 * improvement
        else:
            improvement_term = -0.5 * abs(improvement)

        # component 2: time penalty
        step_penalty = -0.001

        # component 3: success bonus
        success_bonus = 50.0 if (curr_dist < 0.5) else 0.0

        reward = improvement_term + step_penalty + success_bonus

        components = {
            "improvement_term": float(improvement_term),
            "step_penalty": float(step_penalty),
            "success_bonus": float(success_bonus),
            "total_reward": float(reward),
            "improvement": float(improvement),
            "distance": float(curr_dist),
        }
        return float(reward), components

    def step(self, action):
        x, y, z, v, theta = self.state
        dtheta, dz = action

        # normalizza theta in [-π,π]
        theta = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi
        # aggiorna z
        z = z + dz

        x = x + v * np.cos(theta) * self.dt
        y = y + v * np.sin(theta) * self.dt

        prev_state = self.state.copy()
        self.state = np.array([x, y, z, v, theta], dtype=np.float32)
        self.current_step += 1

        reward, comps = self.compute_reward(prev_state, self.state, self.target)

        distance = comps["distance"]
        terminated = distance < 0.5
        truncated = self.current_step >= self.max_steps

        # metriche episodio
        start_distance = float(self._start_distance) if self._start_distance is not None else float("nan")
        total_improvement = start_distance - float(distance)

        info = {
            "distance": float(distance),
            "improvement": float(comps["improvement"]),
            "start_distance": float(start_distance),
            "total_improvement": float(total_improvement),
            "reward_components": {
                "improvement_term": float(comps["improvement_term"]),
                "step_penalty": float(comps["step_penalty"]),
                "success_bonus": float(comps["success_bonus"]),
                "total_reward": float(comps["total_reward"]),
            },
        }

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self, mode="human"):
        print("Pos:", self.state[:3], "Target:", self.target)