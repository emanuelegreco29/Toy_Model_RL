import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import datetime
from collections import deque
import torch.optim as optim

from environments.tag_env import TagEnv
from algorithms.PPO.custom_ppo import CustomActorCritic
from algorithms.utilities.centralized_critic import CentralizedCritic

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
os.makedirs(f"policies/Tag_SP/Tag_SelfPlay_{ts}", exist_ok=True)

# --- Hyperparameters ---
gamma, lam           = 0.99, 0.95
batch_size           = 2048
total_epochs         = 500
save_interval        = 50 # Save every X epochs

clip_eps             = 0.2
entropy_coef         = 0.01
lr_critic            = 5e-5
lr_actor             = 5e-5
K_history            = 1
smoothing_coeff      = 1e-3 # For smoothing the critic loss

# --- Environment ---
env = TagEnv(K_history=K_history)
obs_dict, infos = env.reset()
obs_dim         = env.observation_spaces[env.agents[0]].shape[0]
act_dim         = env.action_spaces   [env.agents[0]].shape[0]
global_state_dim= env.observation_space['global_state'].shape[0]

# --- Centralized critic ---
critic          = CentralizedCritic(global_state_dim)
critic_opt      = optim.Adam(critic.parameters(), lr=lr_critic)

# --- Policy networks and optimizer ---
agents = {}
opts   = {}
for ag in env.agents:
    net = CustomActorCritic(obs_dim, act_dim, log_std_init=-15, trainable_log_std=False)
    net.load_state_dict(torch.load("policies/policy.pth"))
    agents[ag] = net
    opts[ag]   = optim.Adam(net.parameters(), lr=lr_actor)

# --- Training loop ---
for ep in range(1, total_epochs+1):
    buf = {ag: {'obs':[], 'acts':[], 'vals':[], 'logp_old':[], 'rews':[], 'dones':[]} 
           for ag in env.agents}
    batch_global = []

    obs_dict, infos = env.reset()
    g0 = infos[env.agents[0]]['global_state']
    batch_global.append(torch.tensor(g0, dtype=torch.float32))
    steps = 0

    while steps < batch_size:
        actions = {}
        for ag in env.agents:
            o     = torch.tensor(obs_dict[ag], dtype=torch.float32).unsqueeze(0)
            mean, log_std, v = agents[ag](o)
            dist  = torch.distributions.Normal(mean, log_std.exp())
            a     = dist.sample().squeeze().numpy()
            actions[ag] = a

            # Salvo per GAE
            buf[ag]['obs'].append(o.squeeze())
            buf[ag]['acts'].append(torch.tensor(a, dtype=torch.float32))
            buf[ag]['vals'].append(v.squeeze().detach())
            buf[ag]['logp_old'].append(dist.log_prob(torch.tensor(a, dtype=torch.float32)).sum().detach())

        # Step env
        next_obs, rews, dones, infos = env.step(actions)
        g1 = infos[env.agents[0]]['global_state']
        batch_global.append(torch.tensor(g1, dtype=torch.float32))

        for ag in env.agents:
            buf[ag]['rews'].append(torch.tensor(rews[ag], dtype=torch.float32))
            buf[ag]['dones'].append(dones[ag])

        obs_dict = next_obs
        steps += 1

    # Valori dal critic
    values_c   = torch.stack([critic(gs) for gs in batch_global])
    values_det = values_c.detach()

    # Calcolo GAE e returns
    adv_ret = {}
    for ag in env.agents:
        rews_b = torch.stack(buf[ag]['rews'])
        done_b = torch.tensor(buf[ag]['dones'], dtype=torch.float32)
        adv    = torch.zeros_like(rews_b)
        lastgaelam = 0
        for t in reversed(range(len(rews_b))):
            nonterm = 1.0 - done_b[t]
            delta   = rews_b[t] + gamma * values_det[t+1] * nonterm - values_det[t]
            adv[t]  = lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        returns    = adv + values_det[:-1]
        adv        = (adv - adv.mean())/(adv.std()+1e-8)
        adv_ret[ag] = {'adv': adv, 'ret': returns}

    # Update critic
    target = (adv_ret[env.agents[0]]['ret'] + adv_ret[env.agents[1]]['ret']) * 0.5
    critic_loss = ((values_c[:-1] - target)**2).mean()
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # Update policy per agente
    for ag in env.agents:
        obs_b      = torch.stack(buf[ag]['obs'])
        act_b      = torch.stack(buf[ag]['acts'])
        logp_old_b = torch.stack(buf[ag]['logp_old'])
        adv        = adv_ret[ag]['adv']

        act_diff = act_b[1:] - act_b[:-1]
        mean, log_std, _ = agents[ag](obs_b)
        dist  = torch.distributions.Normal(mean, log_std.exp())
        logp_new = dist.log_prob(act_b).sum(-1)
        ratio    = torch.exp(logp_new - logp_old_b)
        s1       = ratio * adv
        s2       = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
        
        policy_loss  = -torch.min(s1, s2).mean()
        entropy_loss = -dist.entropy().sum(-1).mean()
        smooth_loss = smoothing_coeff * (act_diff**2).sum(dim=-1).mean()
        loss = policy_loss + entropy_coef * entropy_loss + smooth_loss

        opts[ag].zero_grad()
        loss.backward()
        opts[ag].step()

    a0, a1 = env.agents
    dist   = infos[a0]['distance']
    fb0    = infos[a0]['followed']
    fb1    = infos[a1]['followed']
    print(f"[Episode {ep}] Distance {dist:.3f} | Behind {a0}: {fb0}, {a1}: {fb1}")
    
    if ep % save_interval == 0:
        for ag in env.agents:
            torch.save(agents[ag].state_dict(),
                    f"policies/Tag_SP/Tag_SelfPlay_{ts}/{ag}_policy_ep{ep}.pth")
        torch.save(critic.state_dict(),
                f"policies/Tag_SP/Tag_SelfPlay_{ts}/critic_ep{ep}.pth")
        print(f"→ Modelli salvati a epoca {ep}")

