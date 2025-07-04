import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import torch
import numpy as np
from environments.dogfight_env import DogfightParallelEnv
from algorithms.PPO.custom_ppo import CustomActorCritic

# --- Hyperparameters and environment ---
env = DogfightParallelEnv(K_history=1)
obs_dict, _ = env.reset()
obs_dim = env.observation_spaces['chaser_0'].shape[0]
act_dim = env.action_spaces['chaser_0'].shape[0]

# --- Define policies ---
chaser = CustomActorCritic(obs_dim, act_dim, log_std_init=-2)
chaser.load_state_dict(torch.load("policies/policy.pth"))
for p in chaser.parameters(): p.requires_grad = False

evader = CustomActorCritic(obs_dim, act_dim, log_std_init=-2)

# --- Optimizers and settings ---
opt_chaser = torch.optim.Adam(chaser.parameters(), lr=3e-4)
opt_evader  = torch.optim.Adam(evader.parameters(),  lr=3e-4)

gamma, lam       = 0.99, 0.95
batch_size       = 500
epochs_per_phase = 100    # epochs per agent phase
cycles           = 4      # number of alternations

def run_phase(train_agent, frozen_agent, optimizer, reward_flip, start_ep):
    ep = start_ep
    train_id = f"{train_agent}_0"
    frozen_id = frozen_agent

    for _ in range(epochs_per_phase):
        # Metrics trackers for this epoch
        episode_rewards = []
        episode_lengths = []
        episode_hps     = []
        curr_reward = 0.0
        curr_length = 0

        # Reset env and batch buffers
        obs_dict, _ = env.reset()
        dones = {a: False for a in env.agents}
        buf_obs, buf_acts, buf_vals, buf_rews, buf_dones, buf_logprobs_old = [], [], [], [], [], []
        steps = 0

        while steps < batch_size:
            actions = {}
            # Sample actions
            for ag in env.agents:
                o = torch.tensor(obs_dict[ag], dtype=torch.float32).unsqueeze(0)
                if ag == frozen_id:
                    net = chaser
                    with torch.no_grad():
                        mean, log_std, _ = net(o)
                        dist = torch.distributions.Normal(mean, log_std.exp())
                        a = dist.sample().squeeze().numpy()
                else:
                    net = evader if train_agent == 'evader' else chaser
                    mean, log_std, v = net(o)
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    a = dist.sample().squeeze().numpy()
                    # store for training
                    buf_obs.append(o.squeeze())
                    buf_acts.append(torch.tensor(a, dtype=torch.float32))
                    buf_vals.append(v.squeeze())
                    lp_old = dist.log_prob(torch.tensor(a, dtype=torch.float32)).sum()
                    buf_logprobs_old.append(lp_old)
                actions[ag] = a

            # Step env
            next_obs, rews, dones, infos = env.step(actions)

            # prendi i termini dal dizionario infos
            info   = infos[train_id]
            shaping= info['shaping']
            wez    = info['wez_step']
            kill   = info['kill']

            # assegna reward diverso per chaser/evader
            if train_agent == 'chaser':
                r = shaping + wez + kill
            else:  # evader vuole allontanarsi (neg. shaping) ma guadagna bonus sparo
                r = -shaping + wez + kill

            buf_rews.append(torch.tensor(r, dtype=torch.float32))
            buf_dones.append(dones[train_id])

            # Update episode metrics
            curr_reward += r
            curr_length += 1

            # Check end of episode for train_agent
            if dones[train_id]:
                # record episode metrics
                episode_rewards.append(curr_reward)
                final_hp = infos[train_id].get('final_hp', 0)
                episode_hps.append(final_hp)
                episode_lengths.append(curr_length)
                
                # reset for next episode
                curr_reward = 0.0
                curr_length = 0
                obs_dict, _ = env.reset()
                dones = {a: False for a in env.agents}
            else:
                obs_dict = next_obs
            steps += 1

        # Compute GAE and returns
        obs_b = torch.stack(buf_obs)
        act_b = torch.stack(buf_acts)
        val_b = torch.stack(buf_vals)
        rew_b = torch.stack(buf_rews)
        done_b= torch.tensor(buf_dones, dtype=torch.float32)
        logprob_old_b = torch.stack(buf_logprobs_old)

        adv = torch.zeros_like(rew_b)
        lastgaelam = 0
        for t in reversed(range(len(rew_b))):
            nonterm = 1.0 - done_b[t]
            next_val = val_b[t+1] if t+1 < len(val_b) else 0
            delta = rew_b[t] + gamma*next_val*nonterm - val_b[t]
            adv[t] = lastgaelam = delta + gamma*lam*nonterm*lastgaelam
        returns = adv + val_b

        # PPO loss and update
        net = evader if train_agent == 'evader' else chaser
        mean, log_std, vals = net(obs_b)
        dist = torch.distributions.Normal(mean, log_std.exp())
        logprob = dist.log_prob(act_b).sum(-1)
        entropy = dist.entropy().sum(-1).mean().item()

        print(f"Entropy: {entropy:.4f}")
        print(f"[Phase {train_agent}] Avg Raw Reward: {torch.stack(buf_rews).mean().item():.4f}")

        # policy_loss = -(logprob * adv.detach()).mean()
        # value_loss  = (returns.detach() - vals.squeeze()).pow(2).mean()
        # loss = policy_loss + 0.5*value_loss - 0.01*entropy

        # --- PPO CLIPPING ---
        clip_eps = 0.2
        # normalizziamo gli advantage
        adv = (adv - adv.mean())/(adv.std() + 1e-8)
        # nuova logprob
        mean, log_std, vals = net(obs_b)
        dist = torch.distributions.Normal(mean, log_std.exp())
        logprob_new = dist.log_prob(act_b).sum(-1)
        # ratio e clipped surrogate
        ratio = torch.exp(logprob_new - logprob_old_b)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        # value loss e entropy rimangono
        value_loss  = (returns.detach() - vals.squeeze()).pow(2).mean()
        entropy_loss = -dist.entropy().sum(-1).mean()
        entropy_coef = 0.1 if train_agent=='chaser' else 0.01
        loss = policy_loss + 0.5*value_loss + entropy_coef*entropy_loss

        print(f"[Adv mean {adv.mean().item():.4f}, std {adv.std().item():.4f}")

        optimizer.zero_grad()
        loss.backward()

        # Logging
        total_norm = 0
        for p in net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Grad Norm: {total_norm:.4f}")

        optimizer.step()

        # Logging each epoch
        avg_reward = np.mean(episode_rewards) if episode_rewards else curr_reward
        avg_hp     = np.mean(episode_hps)     if episode_hps     else infos[train_id].get('final_hp', 0)
        avg_len    = np.mean(episode_lengths) if episode_lengths else curr_length
        
        print(f"[{train_agent} ep {ep}] loss {loss.item():.3f}, ")
        print(f"  avg_reward {avg_reward:.2f}, avg_final_hp {avg_hp:.1f}, avg_len {avg_len:.1f}")

        # Save checkpoint every 50 epochs
        state = evader.state_dict() if train_agent=='evader' else chaser.state_dict()
        if ep % 50 == 0:
            torch.save(state, f"policies/SP/SelfPlay_{ts}/selfplay_{train_agent}_ep{ep}.pth")
        ep += 1

    return ep

# Run alternating training phases
ep_cursor = 0

# Create directories for saving policies
os.makedirs("policies/SP", exist_ok=True)
ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
os.makedirs(f"policies/SP/SelfPlay_{ts}", exist_ok=True)

# Phase 1: evader trains vs frozen chaser
ep_cursor = run_phase('evader', 'chaser_0', opt_evader, reward_flip=False, start_ep=ep_cursor)

# Phase 2: freeze evader, train chaser vs frozen evader
for p in evader.parameters(): p.requires_grad = False
for p in chaser.parameters(): p.requires_grad = True
ep_cursor = run_phase('chaser', 'evader_0', opt_chaser, reward_flip=False, start_ep=ep_cursor)

# Further alternations
for cycle in range(1, cycles):
    # evader phase
    for p in evader.parameters(): p.requires_grad = True
    for p in chaser.parameters(): p.requires_grad = False
    ep_cursor = run_phase('evader', 'chaser_0', opt_evader, reward_flip=False, start_ep=ep_cursor)
    
    # chaser phase
    for p in evader.parameters(): p.requires_grad = False
    for p in chaser.parameters(): p.requires_grad = True
    ep_cursor = run_phase('chaser', 'evader_0', opt_chaser, reward_flip=False, start_ep=ep_cursor)

print("Self-play training completed.")