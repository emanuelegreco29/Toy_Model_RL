import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import torch
import numpy as np
import random
import copy
import torch.optim as optim

from collections import deque
from environments.dogfight_env import DogfightParallelEnv
from algorithms.PPO.custom_ppo import CustomActorCritic
from algorithms.utilities.centralized_critic import CentralizedCritic

# --- Hyperparameters and environment ---
env = DogfightParallelEnv(K_history=1)
obs_dict, _ = env.reset()
obs_dim = env.observation_spaces['chaser_0'].shape[0]
act_dim = env.action_spaces['chaser_0'].shape[0]

# Define the centralized critic
global_state_dim = env.observation_space['global_state'].shape[0]
critic = CentralizedCritic(global_state_dim)
critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

# --- Define policies ---
chaser = CustomActorCritic(obs_dim, act_dim, log_std_init=-2)
chaser.load_state_dict(torch.load("policies/policy.pth"))
for p in chaser.parameters(): p.requires_grad = False

evader = CustomActorCritic(obs_dim, act_dim, log_std_init=-2)
for p in evader.parameters(): p.requires_grad = True

# --- Optimizers and settings ---
opt_chaser = torch.optim.Adam(chaser.parameters(), lr=5e-5)
opt_evader  = torch.optim.Adam(evader.parameters(),  lr=5e-5)

gamma, lam       = 0.99, 0.95
batch_size       = 2048
epochs_per_phase = 10    # epochs per agent phase
cycles           = 30      # number of alternations
pool_size        = 5  # size of the replay pool
clip_eps        = 0.2
entropy_coef    = 0.01

chaser_pool = deque(maxlen=pool_size)
evader_pool = deque(maxlen=pool_size)
chaser_pool.append(copy.deepcopy(chaser))
evader_pool.append(copy.deepcopy(evader))

def run_phase(train_agent, opponent_pool, self_pool, optimizer, start_ep):
    ep = start_ep
    train_id = f"{train_agent}_0"

    for _ in range(epochs_per_phase):
        # sample a frozen opponent snapshot
        opponent_snapshot = random.choice(list(opponent_pool))
        opponent_snapshot.eval()
        for p in opponent_snapshot.parameters():
            p.requires_grad = False

        # metrics for this epoch
        episode_rewards, episode_lengths, episode_hps = [], [], []
        curr_reward, curr_length = 0.0, 0

        # reset env & buffers
        obs_dict, infos = env.reset()
        batch_global_states = []
        batch_global_states.append(torch.tensor(infos[train_id]['global_state'], dtype=torch.float32))
        dones = {a: False for a in env.agents}
        buf_obs, buf_acts, buf_vals = [], [], []
        buf_rews, buf_dones, buf_logp_old = [], [], []
        steps = 0

        while steps < batch_size:
            actions = {}
            for ag in env.agents:
                o = torch.tensor(obs_dict[ag], dtype=torch.float32).unsqueeze(0)

                if ag == train_id:
                    net = evader if train_agent == 'evader' else chaser
                    mean, log_std, v = net(o)
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    a = dist.sample().squeeze().numpy()

                    buf_obs.append(o.squeeze())
                    buf_acts.append(torch.tensor(a, dtype=torch.float32))
                    buf_vals.append(v.squeeze().detach())
                    lp = dist.log_prob(torch.tensor(a, dtype=torch.float32)).sum().detach()
                    buf_logp_old.append(lp)
                else:
                    with torch.no_grad():
                        mean, log_std, _ = opponent_snapshot(o)
                        dist = torch.distributions.Normal(mean, log_std.exp())
                        a = dist.sample().squeeze().numpy()

                actions[ag] = a

            next_obs, rews, dones, infos = env.step(actions)
            batch_global_states.append(torch.tensor(infos[train_id]['global_state'], dtype=torch.float32))
            r = rews[train_id]
            buf_rews.append(torch.tensor(r, dtype=torch.float32))
            buf_dones.append(dones[train_id])

            curr_reward += r
            curr_length += 1

            if dones[train_id]:
                episode_rewards.append(curr_reward)
                episode_hps.append(infos[train_id].get('final_hp', 0))
                episode_lengths.append(curr_length)
                curr_reward, curr_length = 0.0, 0
                obs_dict, _ = env.reset()
                dones = {a: False for a in env.agents}
            else:
                obs_dict = next_obs

            steps += 1

        values_c = torch.stack([critic(gs) for gs in batch_global_states])  # shape [T+1], requires_grad=True

        values_det = values_c.detach()  # shape [T+1]

        # Calcolo GAE & returns
        rew_b   = torch.stack(buf_rews)                             # [T]
        done_b  = torch.tensor(buf_dones, dtype=torch.float32)      # [T]
        adv     = torch.zeros_like(rew_b)
        lastgaelam = 0
        for t in reversed(range(len(rew_b))):
            nonterm = 1.0 - done_b[t]
            delta = rew_b[t] + gamma * values_det[t+1] * nonterm - values_det[t]
            adv[t] = lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        returns = adv + values_det[:-1]   # [T]

        # Update del critic
        critic_loss = ((values_c[:-1] - returns)**2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_b      = torch.stack(buf_obs)
        act_b      = torch.stack(buf_acts)
        logp_old_b = torch.stack(buf_logp_old)

        net = evader if train_agent=='evader' else chaser
        mean, log_std, _ = net(obs_b)
        dist = torch.distributions.Normal(mean, log_std.exp())
        logp_new = dist.log_prob(act_b).sum(-1)

        ratio      = torch.exp(logp_new - logp_old_b)
        surr1      = ratio * adv
        surr2      = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
        policy_loss  = -torch.min(surr1, surr2).mean()
        value_loss   = (returns - values_det[:-1]).pow(2).mean()
        entropy_loss = -dist.entropy().sum(-1).mean()
        loss = policy_loss + 0.5*value_loss + entropy_coef*entropy_loss

        optimizer.zero_grad()
        loss.backward()

        # logging
        total_norm_sq = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm_sq ** 0.5
        entropy = dist.entropy().sum(-1).mean().item()
        print(f"Entropy: {entropy:.4f} | Grad Norm: {total_norm:.4f}")

        optimizer.step()

        if train_agent == 'evader':
            self_pool.append(copy.deepcopy(evader))
        else:
            self_pool.append(copy.deepcopy(chaser))

        # epoch logging & checkpoint
        avg_r = np.mean(episode_rewards) if episode_rewards else curr_reward
        avg_hp = np.mean(episode_hps) if episode_hps else infos[train_id].get('final_hp',0)
        avg_l = np.mean(episode_lengths) if episode_lengths else curr_length
        print(f"[{train_agent} ep {ep}] loss {loss.item():.3f}, "
              f"AvgR {avg_r:.2f} | AvgHP {avg_hp:.1f} | AvgL {avg_l:.1f}")

        if ep % 10 == 0:
            state = evader.state_dict() if train_agent=='evader' else chaser.state_dict()
            torch.save(state, f"policies/SP/SelfPlay_{ts}/selfplay_{train_agent}_ep{ep}.pth")

        ep += 1
    return ep

# Run alternating training phases
ep_cursor = 0

# Create directories for saving policies
os.makedirs("policies/SP", exist_ok=True)
ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
os.makedirs(f"policies/SP/SelfPlay_{ts}", exist_ok=True)

# Further alternations
for cycle in range(cycles):
    # Evader trains against a random chaser snapshot
    for p in evader.parameters(): p.requires_grad = True
    for p in chaser.parameters(): p.requires_grad = False
    ep_cursor = run_phase(
        train_agent='evader',
        opponent_pool=chaser_pool,
        self_pool=evader_pool,
        optimizer=opt_evader,
        start_ep=ep_cursor
    )

    # Chaser trains against a random evader snapshot
    for p in evader.parameters(): p.requires_grad = False
    for p in chaser.parameters(): p.requires_grad = True
    ep_cursor = run_phase(
        train_agent='chaser',
        opponent_pool=evader_pool,
        self_pool=chaser_pool,
        optimizer=opt_chaser,
        start_ep=ep_cursor
    )

print("Self-play training completed.")