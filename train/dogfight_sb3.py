import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from environments.dogfight_env import DogfightParallelEnv

class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.final_hps = []
        self.lengths = []
        self.win_rates = []
    
    def _on_step(self) -> bool:
        # Controlla se ci sono episodi completati
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                ep = info.get("episode")
                if ep:
                    self.rewards.append(ep["r"])
                    final_hp = info.get("final_hp", np.nan)
                    self.final_hps.append(final_hp)
                    self.lengths.append(ep["l"])
                    
                    # Calcola win rate (quando l'HP finale dell'avversario è 0)
                    win = 1 if final_hp == 0 else 0
                    self.win_rates.append(win)
                    
                    n = len(self.rewards)
                    recent_win_rate = np.mean(self.win_rates[-100:]) if len(self.win_rates) >= 100 else np.mean(self.win_rates)
                    print(f"Episode {n:4d} | Reward {ep['r']:+6.2f} | Final HP {final_hp:3.0f} | Len {self.lengths[-1]:3d} | Win Rate {recent_win_rate:.2%}")
        
        return True

class SelfPlayCallback(BaseCallback):
    """
    Custom self-play callback for multi-agent environments.
    Periodically saves the current policy and uses it as opponent.
    """
    
    def __init__(self, save_freq: int = 50000, save_path: str = "selfplay_models", 
                 n_saved_models: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.n_saved_models = n_saved_models
        self.saved_models = []
        self.current_opponent_idx = 0
        
        # Crea directory per salvare i modelli
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Salva il modello corrente
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}.zip")
            self.model.save(model_path)
            
            # Mantieni solo gli ultimi n_saved_models
            self.saved_models.append(model_path)
            if len(self.saved_models) > self.n_saved_models:
                old_model = self.saved_models.pop(0)
                if os.path.exists(old_model):
                    os.remove(old_model)
            
            if self.verbose > 0:
                print(f"Saved model at step {self.n_calls}: {model_path}")
                
        return True

class MultiAgentWrapper(gym.Env):
    """
    Wrapper per gestire l'ambiente multi-agente con SB3.
    Un agente è controllato dalla policy in training, l'altro da una policy fissa.
    """
    
    def __init__(self, base_env_fn, opponent_policy=None):
        super().__init__()
        self.base_env_fn = base_env_fn
        self.env = base_env_fn()
        self.opponent_policy = opponent_policy
        self.last_observations = None
        
        # Determina quale agente controllare (alterneremo tra chaser e evader)
        self.controlled_agent = 'chaser_0'
        self.opponent_agent = 'evader_0'
        
        # Spazi di osservazione e azione per l'agente controllato
        self.observation_space = self.env.observation_spaces[self.controlled_agent]
        self.action_space = self.env.action_spaces[self.controlled_agent]
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Alterna casualmente quale agente controllare
        if np.random.random() < 0.5:
            self.controlled_agent = 'chaser_0'
            self.opponent_agent = 'evader_0'
        else:
            self.controlled_agent = 'evader_0'
            self.opponent_agent = 'chaser_0'
            
        observations, infos = self.env.reset()
        self.last_observations = observations
        return observations[self.controlled_agent], infos.get(self.controlled_agent, {})
    
    def step(self, action):
        # Azione dell'agente controllato
        actions = {self.controlled_agent: action}
        
        # Azione dell'avversario
        if self.opponent_policy is not None:
            obs = self.last_observations[self.opponent_agent]
            opponent_action, _ = self.opponent_policy.predict(obs, deterministic=False)
            actions[self.opponent_agent] = opponent_action
        else:
            # Policy casuale se non c'è opponent
            actions[self.opponent_agent] = self.env.action_spaces[self.opponent_agent].sample()
        
        observations, rewards, dones, infos = self.env.step(actions)
        
        # Salva le osservazioni per il prossimo step
        self.last_observations = observations
        
        # Restituisci solo i dati dell'agente controllato
        obs = observations[self.controlled_agent]
        reward = rewards[self.controlled_agent]
        done = dones[self.controlled_agent]
        terminated = done
        truncated = False
        info = infos[self.controlled_agent]
        
        return obs, reward, terminated, truncated, info
    
    def set_opponent_policy(self, policy):
        self.opponent_policy = policy

def make_env():
    """Crea l'ambiente per il training."""
    return MultiAgentWrapper(lambda: DogfightParallelEnv(K_history=1))

def make_vec_env(n_envs=1):
    """Crea l'ambiente vettorizzato."""
    return DummyVecEnv([make_env for _ in range(n_envs)])

def main():
    # Hyperparametri
    policy_kwargs = dict(
        net_arch=[512, 512, 512],
        log_std_init=-1.5
    )
    
    # Crea ambiente
    env = make_vec_env(n_envs=4)  # Usa 4 ambienti paralleli
    
    # Crea modello PPO
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        batch_size=256,
        n_steps=2048,
        verbose=0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Carica policy pre-allenata se disponibile
    if os.path.exists("policies/policy.pth"):
        try:
            model.policy.load_state_dict(torch.load("policies/policy.pth", map_location=model.device))
            print("Caricata policy pre-allenata da policies/policy.pth")
        except Exception as e:
            print(f"Errore nel caricamento della policy: {e}")
            print("Continuando con policy inizializzata casualmente...")
    
    # Callback per logging e self-play
    logger_cb = EpisodeLogger()
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    selfplay_cb = SelfPlayCallback(
        save_freq=100000,
        save_path=f"policies/SP/SelfPlay_{ts}",
        n_saved_models=5,
        verbose=1
    )
    
    # Training parameters
    total_timesteps = 2_000_000
    selfplay_interval = 200_000  # Ogni quanto cambiare avversario
    
    print("Inizio training con self-play...")
    print(f"Device: {model.device}")
    print(f"Total timesteps: {total_timesteps}")
    
    # Training loop con self-play
    for phase in range(0, total_timesteps, selfplay_interval):
        current_timesteps = min(selfplay_interval, total_timesteps - phase)
        
        print(f"\n=== FASE {phase // selfplay_interval + 1} ===")
        print(f"Training per {current_timesteps} timesteps...")
        
        # Allena per questo intervallo
        model.learn(
            total_timesteps=current_timesteps,
            callback=[logger_cb, selfplay_cb],
            reset_num_timesteps=False
        )
        
        # Aggiorna l'avversario negli ambienti
        if len(selfplay_cb.saved_models) > 0:
            # Carica un modello salvato casualmente come avversario
            opponent_model_path = np.random.choice(selfplay_cb.saved_models)
            try:
                opponent_model = PPO.load(opponent_model_path, device=model.device)
                
                # Aggiorna tutti gli ambienti con il nuovo avversario
                for i in range(env.num_envs):
                    env.envs[i].set_opponent_policy(opponent_model)
                
                print(f"Aggiornato avversario con: {opponent_model_path}")
            except Exception as e:
                print(f"Errore nel caricamento del modello avversario: {e}")
    
    # Plotting dei risultati
    if len(logger_cb.rewards) > 0:
        eps = np.arange(1, len(logger_cb.rewards) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward
        ax1.plot(eps, logger_cb.rewards, alpha=0.3)
        if len(logger_cb.rewards) > 100:
            # Media mobile
            window = 100
            moving_avg = np.convolve(logger_cb.rewards, np.ones(window)/window, mode='valid')
            ax1.plot(eps[window-1:], moving_avg, 'r-', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.grid(True)
        
        # Final HP
        ax2.plot(eps, logger_cb.final_hps, alpha=0.3)
        if len(logger_cb.final_hps) > 100:
            moving_avg = np.convolve(logger_cb.final_hps, np.ones(window)/window, mode='valid')
            ax2.plot(eps[window-1:], moving_avg, 'r-', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Final HP')
        ax2.set_title('Final HP of Opponent')
        ax2.grid(True)
        
        # Episode length
        ax3.plot(eps, logger_cb.lengths, alpha=0.3)
        if len(logger_cb.lengths) > 100:
            moving_avg = np.convolve(logger_cb.lengths, np.ones(window)/window, mode='valid')
            ax3.plot(eps[window-1:], moving_avg, 'r-', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Episode Lengths')
        ax3.grid(True)
        
        # Win rate
        if len(logger_cb.win_rates) > 100:
            win_rate_smooth = np.convolve(logger_cb.win_rates, np.ones(window)/window, mode='valid')
            ax4.plot(eps[window-1:], win_rate_smooth, 'g-', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Win Rate')
        ax4.set_title('Win Rate (100-episode moving average)')
        ax4.set_ylim(0, 1)
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()
    
    # Salva il modello finale
    os.makedirs('models', exist_ok=True)
    final_model_path = f"models/ppo_selfplay_{ts}.zip"
    model.save(final_model_path)
    
    print(f"\nSelf-play training completato!")
    print(f"Modello salvato in: {final_model_path}")
    print(f"Episodi totali: {len(logger_cb.rewards)}")
    if len(logger_cb.win_rates) > 0:
        print(f"Win rate finale: {np.mean(logger_cb.win_rates[-100:]):.2%}")

if __name__ == "__main__":
    main()