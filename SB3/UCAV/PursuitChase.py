import numpy as np
import gymnasium as gym
from gymnasium import spaces

from UCAV import UCAV
from Prey import StraightLineAttacker

class ChaseEnv(gym.Env):
    def __init__(self):
        self.dt       = 0.1
        self.step_count = 0
        self.max_steps = 500
        self.ucav_env = UCAV(dt=self.dt)
        self.attacker = StraightLineAttacker(v=200.0, dt=self.dt)

        # action & obs spazio uguali a quelli di UCAV
        self.action_space      = self.ucav_env.action_space
        self.observation_space = self.ucav_env.observation_space
        self.reset()

    def reset(self, seed=None):
        self.step_count = 0

        # reset UCAV e attacker
        obs_ucav = self.ucav_env.reset()
        pos_t, vel_t = self.attacker.reset()

        # aggiorna target
        self.ucav_env.target_position = pos_t
        self.ucav_env.target_velocity = vel_t
        return obs_ucav, {}

    def step(self, action):
        self.step_count += 1

        # primo muovo e aggiorno il target
        pos_t, vel_t = self.attacker.step()
        self.ucav_env.target_position = pos_t
        self.ucav_env.target_velocity = vel_t

        # azione UCAV
        obs, _, _, info = self.ucav_env.step(action)

        # calcolo la reward complessiva
        reward = self.ucav_env.total_reward(obs, self.ucav_env.reward_params)
        R, _, ATA, AA, _, _, _, z_U, v_U = obs
        p = self.ucav_env.reward_params
        success = (R <= p['R_w'] and ATA < p['phi_w'] and AA < p['q_w'])
        
        # failure condition
        fail = (R < p['R_min'] or R > p['R_max']
                or z_U < p['z_min'] or z_U > p['z_max']
                or v_U < p['v_min'] or v_U > p['v_max'])
        
        info = {
            "is_success": success,
            "is_failure": fail
        }

        terminated = bool(success or fail)
        truncated  = (self.step_count >= self.max_steps)
        return obs, reward, terminated, truncated, info