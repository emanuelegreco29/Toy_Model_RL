import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from algorithms.PPO.ppo_policy import PPOPolicy
from algorithms.PPO.ppo_trainer import PPOTrainer
from algorithms.utilities.buffer import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.bezier_history import PointMassEnv as BezierEnv

class EpisodeLogger:
    def __init__(self):
        self.rewards = []
        self.distances = []
        self.followed = []
        
    def __call__(self, locals_, globals_):
        for info in locals_["infos"]:
            ep = info.get("episode")
            if ep:
                r = ep["r"]
                d = info.get("distance", np.nan)
                f = info.get("followed", np.nan)
                n = len(self.rewards) + 1
                self.rewards.append(r)
                self.distances.append(d)
                self.followed.append(f)
                print(f"Episode {n} | Reward: {r:.2f} | Followed for: {f}")
        return True
    

class Args:
    lr = 1e-4
    clip_param = 0.2
    ppo_epoch = 10
    num_mini_batch = 8
    value_loss_coef = 0.5
    entropy_coef = 0.02
    use_max_grad_norm = True
    max_grad_norm = 0.5
    use_clipped_value_loss = True
    use_feature_normalization = True
    use_recurrent_policy = False
    recurrent_hidden_size = 128
    recurrent_hidden_layers = 1
    data_chunk_length = 1
    gamma = 0.99
    gae_lambda = 0.95
    buffer_size = 2048
    n_rollout_threads = 1

    # architettura rete
    hidden_size = "64 64"
    act_hidden_size = "64"
    activation_id = 0
    gain = 1.0

args = Args()

# Choose the environment to train on    
print("Choose the environment to train on: \n1. Bezier trajectory")
choice = input("Environment: ")
if choice != '1':
    raise ValueError("Invalid choice. Only option 1 is available.")
else:
    chosen_env = 'bezier_history'

# Choose the reward function
print("Choose the reward function: \n1. Predictive reward")
choice = input("Reward function: ")
if choice != '1':
    raise ValueError("Invalid choice. Only option 1 is available.")
else:
    chosen_reward = 'predictive'

# Choose the algorithm to train
#print("Choose the algorithm to train: \n1. PPO")
#choice = input("Algorithm: ")
#if choice != '1':
#    raise ValueError("Invalid choice. Only option 1 is available.")
#else:
#    chosen_algorithm = 'ppo'

# Choose the number of episodes
print("Choose the number of episodes to train.\n")
episodes_number = input("Episodes: ")
episodes_number = int(episodes_number)

if chosen_env == 'bezier_history':
    env = DummyVecEnv([lambda: Monitor(BezierEnv(reward_fn=chosen_reward))])

obs_space = env.observation_space
act_space = env.action_space

policy = PPOPolicy(args, obs_space, act_space, device="cpu")
trainer = PPOTrainer(args, device="cpu")
buffer  = ReplayBuffer(args, num_agents=1, obs_space=obs_space, act_space=act_space)

callback = EpisodeLogger()
total_timesteps = 500 * episodes_number
num_updates = total_timesteps // (args.buffer_size * args.n_rollout_threads)

obs = env.reset()
rnn_states_actor  = np.zeros((1, args.recurrent_hidden_layers, args.recurrent_hidden_size), dtype=np.float32)
rnn_states_critic = np.zeros_like(rnn_states_actor)
masks = np.ones((1, 1), dtype=np.float32)


for update in range(num_updates):
    for step in range(args.buffer_size):
        # 1. seleziona azione
        with torch.no_grad():
            values, actions, log_probs, rnn_states_actor, rnn_states_critic = \
                policy.get_actions(obs, rnn_states_actor, rnn_states_critic, masks)

        # 2. esegui step nell’ambiente
        next_obs, reward, done, info = env.step(actions[0])
        mask = 0.0 if done else 1.0

        # 3. inserisci dati nel buffer
        buffer.insert(
            obs=np.expand_dims(obs, 0),
            actions=np.expand_dims(actions, 0),
            rewards=np.array([[reward]], dtype=np.float32),
            masks=np.array([[mask]], dtype=np.float32),
            action_log_probs=np.expand_dims(log_probs, 0),
            value_preds=np.expand_dims(values, 0),
            rnn_states_actor=rnn_states_actor,
            rnn_states_critic=rnn_states_critic
        )

        obs = next_obs
        masks = np.array([[mask]], dtype=np.float32)

    # calcola il valore dell’ultimo stato per GAE
    with torch.no_grad():
        next_value = policy.get_values(obs[np.newaxis, :], rnn_states_critic, masks)

    buffer.compute_returns(next_value)  # metodo interno di ReplayBuffer

    # aggiorna la policy con mini‐batch ricorrenti
    for sample in buffer.recurrent_generator(buffer, args.num_mini_batch, args.data_chunk_length):
        trainer.ppo_update(policy, sample)

    buffer.after_update()
    policy.optimizer.step()

eps = np.arange(1, len(callback.rewards) + 1)
fig, axs = plt.subplots(2,1, figsize=(8,6))
axs[0].plot(eps, callback.rewards, marker='o')
axs[0].set(xlabel='Episode', ylabel='Reward')
axs[1].plot(eps, callback.followed, marker='o', color='orange')
axs[1].set(xlabel='Episode', ylabel='Followed target for X steps')
plt.tight_layout()
plt.show()

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs('models', exist_ok=True)
torch.save({
    'actor_state_dict':  policy.actor.state_dict(),
    'critic_state_dict': policy.critic.state_dict(),
    'args': vars(args)  # utile per ricreare la stessa configurazione
}, f'models/ppo_full_{ts}.pt')
print("\nTraining completed!")