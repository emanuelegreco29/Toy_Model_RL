import os
import sys

# ensure relative imports work when running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import math
import datetime
import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from environments.tag_shoot_env import TagShootEnv
from algorithms.IPPO.models import EnhancedActorCritic
from algorithms.IPPO.utils import set_seed, safecpu, explained_variance
from algorithms.IPPO.pfsp_utils import OpponentPool, other

# Improved hyperparameters
total_timesteps = 1000000
steps_per_update = 5000
update_epochs = 4
num_minibatches = 16         # More minibatches for better gradient estimates
learning_rate = 1e-4
gamma = 0.995                # Higher discount for longer-term planning
gae_lambda = 0.95
clip_coef = 0.2
ent_coef = 0.03              # Lower entropy for more focused policies
vf_coef = 0.5
max_grad_norm = 0.5
seed = 42
device = "cpu"
save_interval = 25
log_interval = 1
tag = "Tag_Shoot_IPPO"
warmup_updates = 10 # For LR scheduler
ent_start = 0.08
ent_end   = 0.004

# Switches
PFSP_ENABLED = False

def lr_lambda_fn(update_idx):
    if update_idx < warmup_updates:
        return float(update_idx + 1) / float(warmup_updates)  # 0->1
    # progress 0..1 sul resto degli update
    prog = (update_idx - warmup_updates) / max(1, (total_timesteps//steps_per_update - warmup_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
    return 0.1 + 0.9 * cosine  # 1.0 -> 0.1

def entropy_coef(update_idx):
    total_updates = total_timesteps // steps_per_update
    prog = min(1.0, update_idx / max(1, total_updates))
    return ent_start * (1.0 - prog) + ent_end * prog

def _default_metrics():
    return {
        "ev": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "ratio_mean": 1.0,
        "ratio_std": 0.0,
        "lr": learning_rate,
    }

# ---------- Setup ----------
set_seed(seed)
device = torch.device(device)

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
save_dir = os.path.join("policies", "Tag_Shoot_IPPO", f"{tag}_{ts}")
os.makedirs(save_dir, exist_ok=True)

# Save hyperparameters in a logging file
with open(os.path.join(save_dir, "hyperparams.txt"), "w") as f:
    f.write(f"total_timesteps: {total_timesteps}\n")
    f.write(f"steps_per_update: {steps_per_update}\n")
    f.write(f"update_epochs: {update_epochs}\n")
    f.write(f"num_minibatches: {num_minibatches}\n")
    f.write(f"learning_rate: {learning_rate}\n")
    f.write(f"gamma: {gamma}\n")
    f.write(f"gae_lambda: {gae_lambda}\n")
    f.write(f"clip_coef: {clip_coef}\n")
    f.write(f"ent_coef: {ent_coef}\n")
    f.write(f"vf_coef: {vf_coef}\n")

env = TagShootEnv()
agent_names = env.agents
obs_dim = env.observation_spaces[agent_names[0]].shape[0]
act_dim = env.action_spaces[agent_names[0]].shape[0]

print(f"Observation dimension: {obs_dim}")
print(f"Action dimension: {act_dim}")

# --- Shared actor + per-agent critics ---
nets = {ag: EnhancedActorCritic(obs_dim, act_dim).to(device) for ag in agent_names}

# Same actor for both agents
shared_actor = nets[agent_names[0]].actor
nets[agent_names[1]].actor = shared_actor

# Single optimizer for shared actor
actor_opt = Adam(shared_actor.parameters(), lr=learning_rate, eps=1e-5)

# Two optimizers for the separate critics
crit_opts = {ag: Adam(nets[ag].critic.parameters(), lr=learning_rate, eps=1e-5) for ag in agent_names}

# PFSP
if PFSP_ENABLED:
    pools = {ag: OpponentPool(ag, obs_dim, act_dim, device) for ag in agent_names}
    # Set random policies for start
    for ag in agent_names:
        pools[ag].add(nets[ag].state_dict())
else:
    pools = None

metrics = {ag: _default_metrics() for ag in agent_names}

# Separate LearningRate schedulers
actor_sched = LambdaLR(actor_opt, lr_lambda=lr_lambda_fn)
crit_sched  = {ag: LambdaLR(crit_opts[ag], lr_lambda=lr_lambda_fn) for ag in agent_names}

# Rollout buffers per-agent
buf = {}
for ag in agent_names:
    buf[ag] = {
        "obs": torch.zeros((steps_per_update, obs_dim), dtype=torch.float32, device=device),
        "acts": torch.zeros((steps_per_update, act_dim), dtype=torch.float32, device=device),
        "logp": torch.zeros((steps_per_update,), dtype=torch.float32, device=device),
        "rews": torch.zeros((steps_per_update,), dtype=torch.float32, device=device),
        "dones": torch.zeros((steps_per_update,), dtype=torch.float32, device=device),
        "vals": torch.zeros((steps_per_update,), dtype=torch.float32, device=device),
        "advantages": torch.zeros((steps_per_update,), dtype=torch.float32, device=device),
        "returns": torch.zeros((steps_per_update,), dtype=torch.float32, device=device),
    }

# Metrics tracking
episode_returns = {ag: [] for ag in agent_names}
episode_lengths = {ag: [] for ag in agent_names}
episode_hits = {ag: [] for ag in agent_names}

global_step = 0
start_time = time.time()

num_updates = total_timesteps // steps_per_update
obs_dict, infos = env.reset()
next_obs = {ag: torch.tensor(obs_dict[ag], dtype=torch.float32, device=device) for ag in agent_names}
next_done = {ag: torch.zeros((), dtype=torch.float32, device=device) for ag in agent_names}

print(f"Starting training: {num_updates} updates, {total_timesteps} total timesteps")
print(f"Device: {device}")

# Training loop
for update in range(1, num_updates + 1):
    ent_coef_now = entropy_coef(update)
    total_updates = total_timesteps // steps_per_update

    # accumuli per logging
    ep_returns_batch = {ag: [] for ag in agent_names}
    ep_lens_batch    = {ag: [] for ag in agent_names}
    ep_hits_batch    = {ag: [] for ag in agent_names}
    ep_ret_acc = {ag: 0.0 for ag in agent_names}
    ep_len_acc = {ag: 0   for ag in agent_names}
    ep_hit_acc = {ag: 0   for ag in agent_names}

    if PFSP_ENABLED:
        train_agent = agent_names[update % 2]
        opp_agent   = agent_names[1 - (update % 2)]
        opp_idx, opp_actor = pools[opp_agent].sample_actor()
        for step in range(steps_per_update):
            global_step += 1
            # store obs/dones
            for ag in agent_names:
                buf[ag]["obs"][step]   = next_obs[ag]
                buf[ag]["dones"][step] = next_done[ag]
            # azioni: train usa rete corrente; opp usa snapshot PFSP (o rete corrente fallback)
            with torch.no_grad():
                obs_noise_tr = torch.randn_like(next_obs[train_agent]) * 0.01
                a_tr, logp_tr, _, v_tr = nets[train_agent].get_action_and_value(
                    (next_obs[train_agent] + obs_noise_tr).unsqueeze(0)
                )
                a_tr, logp_tr, v_tr = a_tr.squeeze(0), logp_tr.squeeze(0), v_tr.squeeze(0)
                opponent_policy = opp_actor if opp_actor is not None else nets[opp_agent]
                a_op, _, _, _ = opponent_policy.get_action_and_value(next_obs[opp_agent].unsqueeze(0))
                a_op = a_op.squeeze(0)

                # clip
                low_tr  = torch.as_tensor(env.action_spaces[train_agent].low,  device=device, dtype=torch.float32)
                high_tr = torch.as_tensor(env.action_spaces[train_agent].high, device=device, dtype=torch.float32)
                a_tr = torch.clamp(a_tr, low_tr, high_tr)
                low_op  = torch.as_tensor(env.action_spaces[opp_agent].low,  device=device, dtype=torch.float32)
                high_op = torch.as_tensor(env.action_spaces[opp_agent].high, device=device, dtype=torch.float32)
                a_op = torch.clamp(a_op, low_op, high_op)

                # buffer SOLO lato allenato
                buf[train_agent]["acts"][step] = a_tr
                buf[train_agent]["logp"][step] = logp_tr
                buf[train_agent]["vals"][step] = v_tr

            # step env
            action_np = {train_agent: safecpu(a_tr), opp_agent: safecpu(a_op)}
            next_obs_np, rewards, dones, infos = env.step(action_np)

            # salva reward/done SOLO per lato allenato
            r = float(rewards[train_agent])
            d = float(dones[train_agent])
            buf[train_agent]["rews"][step] = r
            next_done[train_agent] = torch.tensor(d, device=device, dtype=torch.float32)

            # accumuli per logging (solo lato allenato)
            ep_ret_acc[train_agent] += r
            ep_len_acc[train_agent] += 1
            ep_hit_acc[train_agent]  = infos[train_agent].get('Episode Hits', 0)

            # reset se finito
            next_obs = {ag: torch.tensor(next_obs_np[ag], dtype=torch.float32, device=device) for ag in agent_names}
            if dones["__all__"]:
                hits_tr = infos[train_agent].get('Episode Hits', 0)
                hits_op = infos[opp_agent].get('Episode Hits', 0)
                hp_tr   = infos[train_agent].get(f'{train_agent} HP', 100)
                hp_op   = infos[train_agent].get(f'{opp_agent} HP', 100)
                win = (hits_tr > hits_op) or (hits_tr == hits_op and hp_tr > hp_op)
                pools[opp_agent].record_result(opp_idx, win)

                ep_returns_batch[train_agent].append(ep_ret_acc[train_agent])
                ep_lens_batch[train_agent].append(ep_len_acc[train_agent])
                ep_hits_batch[train_agent].append(hits_tr)
                ep_ret_acc[train_agent] = 0.0
                ep_len_acc[train_agent] = 0
                ep_hit_acc[train_agent] = 0

                obs_dict, infos = env.reset()
                next_obs  = {ag: torch.tensor(obs_dict[ag], dtype=torch.float32, device=device) for ag in agent_names}
                next_done = {ag: torch.zeros((), dtype=torch.float32, device=device) for ag in agent_names}

    else:
        for step in range(steps_per_update):
            global_step += 1
            for ag in agent_names:
                buf[ag]["obs"][step]   = next_obs[ag]
                buf[ag]["dones"][step] = next_done[ag]

            with torch.no_grad():
                actions = {}
                for ag in agent_names:
                    obs_noise = torch.randn_like(next_obs[ag]) * 0.01
                    a, logp, _, v = nets[ag].get_action_and_value((next_obs[ag] + obs_noise).unsqueeze(0))
                    a, logp, v = a.squeeze(0), logp.squeeze(0), v.squeeze(0)
                    # clip
                    low  = torch.as_tensor(env.action_spaces[ag].low,  device=device, dtype=torch.float32)
                    high = torch.as_tensor(env.action_spaces[ag].high, device=device, dtype=torch.float32)
                    a = torch.clamp(a, low, high)
                    # store in buffer per entrambi
                    buf[ag]["acts"][step] = a
                    buf[ag]["logp"][step] = logp
                    buf[ag]["vals"][step] = v
                    actions[ag] = a

            # step env con entrambe le azioni correnti
            action_np = {ag: safecpu(actions[ag]) for ag in agent_names}
            next_obs_np, rewards, dones, infos = env.step(action_np)

            # salva reward/done per entrambi
            for ag in agent_names:
                r = float(rewards[ag])
                d = float(dones[ag])
                buf[ag]["rews"][step]  = r
                next_done[ag] = torch.tensor(d, device=device, dtype=torch.float32)
                ep_ret_acc[ag] += r
                ep_len_acc[ag] += 1
                ep_hit_acc[ag]  = infos[ag].get('Episode Hits', 0)

            next_obs = {ag: torch.tensor(next_obs_np[ag], dtype=torch.float32, device=device) for ag in agent_names}
            if dones["__all__"]:
                for ag in agent_names:
                    ep_returns_batch[ag].append(ep_ret_acc[ag])
                    ep_lens_batch[ag].append(ep_len_acc[ag])
                    ep_hits_batch[ag].append(infos[ag].get('Episode Hits', 0))
                    ep_ret_acc[ag] = 0.0
                    ep_len_acc[ag] = 0
                    ep_hit_acc[ag] = 0
                obs_dict, infos = env.reset()
                next_obs  = {ag: torch.tensor(obs_dict[ag], dtype=torch.float32, device=device) for ag in agent_names}
                next_done = {ag: torch.zeros((), dtype=torch.float32, device=device) for ag in agent_names}

    # Store episode stats
    for ag in agent_names:
        episode_returns[ag].extend(ep_returns_batch[ag])
        episode_lengths[ag].extend(ep_lens_batch[ag])
        episode_hits[ag].extend(ep_hits_batch[ag])
        
    # ===== GAE & UPDATE =====
    targets = [agent_names[update % 2]] if PFSP_ENABLED else agent_names

    # GAE per tutti i lati target
    for ag in targets:
        with torch.no_grad():
            next_value = nets[ag].get_value(next_obs[ag].unsqueeze(0)).squeeze(0)
        lastgaelam = 0.0
        for t in reversed(range(steps_per_update)):
            if t == steps_per_update - 1:
                next_nonterminal = 1.0 - float(next_done[ag].item())
                next_val = next_value
            else:
                next_nonterminal = 1.0 - float(buf[ag]["dones"][t + 1].item())
                next_val = buf[ag]["vals"][t + 1]
            delta = buf[ag]["rews"][t] + gamma * next_val * next_nonterminal - buf[ag]["vals"][t]
            lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
            buf[ag]["advantages"][t] = lastgaelam
        buf[ag]["returns"] = buf[ag]["advantages"] + buf[ag]["vals"]
        adv = buf[ag]["advantages"]
        buf[ag]["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    # UPDATE (actor condiviso + critic per-agente)
    bsz = steps_per_update
    minibatch_size = bsz // num_minibatches

    for ag in targets:
        policy_losses, value_losses, entropy_losses, ratios = [], [], [], []

    inds = np.arange(bsz)
    for epoch in range(update_epochs):
        np.random.shuffle(inds)
        for start in range(0, bsz, minibatch_size):
            mb = inds[start:start + minibatch_size]

            # azzera grad dell'actor condiviso; i critic si azzerano per-agente
            actor_opt.zero_grad(set_to_none=True)
            batch_ratios = []

            for ag in targets:
                obs_b      = buf[ag]["obs"][mb]
                acts_b     = buf[ag]["acts"][mb]
                logp_old_b = buf[ag]["logp"][mb]
                adv_b      = buf[ag]["advantages"][mb]
                ret_b      = buf[ag]["returns"][mb]
                val_b      = buf[ag]["vals"][mb]

                # fwd: usa actor condiviso (legato dentro nets[ag]) + critic specifico
                _, logp, entropy, value = nets[ag].get_action_and_value(obs_b, acts_b)
                ratio = (logp - logp_old_b).exp()
                batch_ratios.append(ratio.detach().cpu().numpy())

                # PPO actor loss
                pg1 = -adv_b * ratio
                pg2 = -adv_b * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                # value loss (critic per-agente)
                v_uncl = (value - ret_b) ** 2
                v_clip = val_b + torch.clamp(value - val_b, -clip_coef, clip_coef)
                v_loss = 0.5 * torch.max(v_uncl, (v_clip - ret_b) ** 2).mean()

                actor_loss  = pg_loss - entropy_coef(update) * entropy.mean()
                critic_loss = vf_coef * v_loss

                # backward critic (step immediato per-agente)
                crit_opts[ag].zero_grad(set_to_none=True)
                critic_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(nets[ag].critic.parameters(), max_grad_norm)
                crit_opts[ag].step()

                # accumula grad dell'actor condiviso (niente step qui)
                actor_loss.backward(retain_graph=True)

                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(float(entropy.mean().item()))

            # singolo step dell'actor condiviso per questo minibatch
            torch.nn.utils.clip_grad_norm_(shared_actor.parameters(), max_grad_norm)
            actor_opt.step()
            for arr in batch_ratios:
                ratios.extend(arr)

    # step schedulers
    actor_sched.step()
    for ag in targets:
        crit_sched[ag].step()

    # explained variance / metrics
    for ag in targets:
        with torch.no_grad():
            v_pred = safecpu(nets[ag].get_value(buf[ag]["obs"]).cpu())
            v_true = safecpu(buf[ag]["returns"].cpu())
            ev = explained_variance(v_pred, v_true)
        metrics[ag] = {
            "ev": ev,
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std": float(np.std(ratios)),
            "lr": actor_sched.get_last_lr()[0],  # lr dell'actor condiviso
        }
        
    # LOGGING AND SAVING
    if update % log_interval == 0:
        # Compute episode statistics
        stats_msg = []
        for ag in agent_names:
            if len(ep_returns_batch[ag]) > 0:
                r_mean = float(np.mean(ep_returns_batch[ag]))
                r_std = float(np.std(ep_returns_batch[ag]))
                l_mean = float(np.mean(ep_lens_batch[ag]))
                h_mean = float(np.mean(ep_hits_batch[ag])) if len(ep_hits_batch[ag]) > 0 else 0.0
            else:
                r_mean = float(ep_ret_acc[ag])
                r_std = 0.0
                l_mean = float(ep_len_acc[ag])
                h_mean = 0.0
                
            stats_msg.append(
                f"[{ag}] R:{r_mean:+.2f}±{r_std:.2f} L:{l_mean:.0f} H:{h_mean:.1f} "
                f"EV:{metrics[ag]['ev']:.2f} π:{metrics[ag]['policy_loss']:.3f} "
                f"V:{metrics[ag]['value_loss']:.3f} S:{metrics[ag]['entropy']:.3f} "
                f"lr:{metrics[ag]['lr']:.1e}"
            )

        elapsed = time.time() - start_time
        fps = int(global_step / elapsed)
        
        print(f"Update {update:4d}/{num_updates} | Steps {global_step:7d} | FPS {fps:4d}")
        for msg in stats_msg:
            print(f"  {msg}")
        print()

    # Save checkpoints
    if update % save_interval == 0:
        for ag in agent_names:
            path = os.path.join(save_dir, f"{ag}_policy_upd{update:04d}.pth")
            torch.save({
                'model_state_dict': nets[ag].state_dict(),
                'actor_opt_state_dict': actor_opt.state_dict(),
                'critic_opt_state_dict': crit_opts[ag].state_dict(),
                'actor_sched_state_dict': actor_sched.state_dict(),
                'critic_sched_state_dict': crit_sched[ag].state_dict(),
                'update': update,
                'global_step': global_step,
                'metrics': metrics[ag]
            }, path)
            if PFSP_ENABLED:
                pools[ag].add(nets[ag].state_dict())
                
        # Save training statistics
        stats_path = os.path.join(save_dir, f"training_stats_upd{update:04d}.npz")
        np.savez(stats_path,
                 episode_returns_0=episode_returns[agent_names[0]],
                 episode_returns_1=episode_returns[agent_names[1]],
                 episode_lengths_0=episode_lengths[agent_names[0]],
                 episode_lengths_1=episode_lengths[agent_names[1]],
                 episode_hits_0=episode_hits[agent_names[0]],
                 episode_hits_1=episode_hits[agent_names[1]])
        
        print(f"Saved checkpoints and stats at update {update} in {save_dir}")

# Final save
print("Training completed!")
for ag in agent_names:
    final_path = os.path.join(save_dir, f"{ag}_policy_final.pth")
    torch.save({
        'model_state_dict': nets[ag].state_dict(),
        'actor_opt_state_dict': actor_opt.state_dict(),
        'critic_opt_state_dict': crit_opts[ag].state_dict(),
        'actor_sched_state_dict': actor_sched.state_dict(),
        'critic_sched_state_dict': crit_sched[ag].state_dict(),
        'update': update,
        'global_step': global_step,
        'final': True
    }, final_path)
    
print(f"Final models saved in {save_dir}")

# Print final statistics
print("\n=== FINAL TRAINING STATISTICS ===")
for ag in agent_names:
    if len(episode_returns[ag]) > 0:
        returns = episode_returns[ag]
        lengths = episode_lengths[ag]
        hits = episode_hits[ag]
        print(f"{ag}:")
        print(f"  Episodes: {len(returns)}")
        print(f"  Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"  Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"  Hits/Episode: {np.mean(hits):.2f} ± {np.std(hits):.2f}")
        print(f"  Best Episode Return: {np.max(returns):.2f}")
        print()