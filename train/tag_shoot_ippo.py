import os
import sys

# ensure relative imports work when running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import datetime
import numpy as np
import torch

from torch.optim import Adam
from environments.tag_shoot_env import TagShootEnv
from algorithms.IPPO.models import EnhancedActorCritic
from algorithms.IPPO.utils import set_seed, safecpu, explained_variance

# Improved hyperparameters
total_timesteps = 2000000
steps_per_update = 4096
update_epochs = 4
num_minibatches = 16         # More minibatches for better gradient estimates
learning_rate = 1e-4
gamma = 0.995                # Higher discount for longer-term planning
gae_lambda = 0.95
clip_coef = 0.2
ent_coef = 0.01              # Lower entropy for more focused policies
vf_coef = 0.5
max_grad_norm = 0.5
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
save_interval = 25
k_history = 5                # More history for better tactical awareness
log_interval = 1
tag = "Tag_Shoot_IPPO"

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
    f.write(f"k_history: {k_history}\n")

env = TagShootEnv(K_history=k_history)
agent_names = env.agents
obs_dim = env.observation_spaces[agent_names[0]].shape[0]
act_dim = env.action_spaces[agent_names[0]].shape[0]

print(f"Observation dimension: {obs_dim}")
print(f"Action dimension: {act_dim}")

# Independent PPO networks and optimizers per-agent
nets = {ag: EnhancedActorCritic(obs_dim, act_dim).to(device) for ag in agent_names}
opts = {ag: Adam(nets[ag].parameters(), lr=learning_rate, eps=1e-5) for ag in agent_names}

# Learning rate schedulers
schedulers = {ag: torch.optim.lr_scheduler.LinearLR(opts[ag], start_factor=1.0, end_factor=0.1, 
                                                   total_iters=total_timesteps//steps_per_update) 
              for ag in agent_names}

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
    # Collect rollout data
    ep_returns_batch = {ag: [] for ag in agent_names}
    ep_lens_batch = {ag: [] for ag in agent_names}
    ep_hits_batch = {ag: [] for ag in agent_names}
    ep_ret_acc = {ag: 0.0 for ag in agent_names}
    ep_len_acc = {ag: 0 for ag in agent_names}
    ep_hit_acc = {ag: 0 for ag in agent_names}

    for step in range(steps_per_update):
        global_step += 1

        # Store current obs and dones
        for ag in agent_names:
            buf[ag]["obs"][step] = next_obs[ag]
            buf[ag]["dones"][step] = next_done[ag]

        # Get actions from policies
        with torch.no_grad():
            actions = {}
            logps = {}
            values = {}
            for ag in agent_names:
                # Add noise to observations during training for exploration
                obs_noise = torch.randn_like(next_obs[ag]) * 0.01
                noisy_obs = next_obs[ag] + obs_noise
                
                a, logp, _, v = nets[ag].get_action_and_value(noisy_obs.unsqueeze(0))
                a = a.squeeze(0)
                logp = logp.squeeze(0)
                v = v.squeeze(0)
                
                # Clip to action space
                low = torch.as_tensor(env.action_spaces[ag].low, device=device, dtype=torch.float32)
                high = torch.as_tensor(env.action_spaces[ag].high, device=device, dtype=torch.float32)
                a = torch.clamp(a, low, high)
                
                actions[ag] = a
                logps[ag] = logp
                values[ag] = v

        # Save to buffer
        for ag in agent_names:
            buf[ag]["acts"][step] = actions[ag]
            buf[ag]["logp"][step] = logps[ag]
            buf[ag]["vals"][step] = values[ag]

        # Environment step
        action_np = {ag: safecpu(actions[ag]) for ag in agent_names}
        next_obs_np, rewards, dones, infos = env.step(action_np)

        # Store rewards and dones
        for ag in agent_names:
            r = float(rewards[ag])
            done = float(dones[ag])
            buf[ag]["rews"][step] = r
            next_done[ag] = torch.tensor(done, device=device, dtype=torch.float32)

            ep_ret_acc[ag] += r
            ep_len_acc[ag] += 1
            ep_hit_acc[ag] = infos[ag].get('Episode Hits', 0)
            
            if done:
                ep_returns_batch[ag].append(ep_ret_acc[ag])
                ep_lens_batch[ag].append(ep_len_acc[ag])
                ep_hits_batch[ag].append(ep_hit_acc[ag])
                
                ep_ret_acc[ag] = 0.0
                ep_len_acc[ag] = 0
                ep_hit_acc[ag] = 0

        next_obs = {ag: torch.tensor(next_obs_np[ag], dtype=torch.float32, device=device) for ag in agent_names}

        if dones["__all__"]:
            obs_dict, infos = env.reset()
            next_obs = {ag: torch.tensor(obs_dict[ag], dtype=torch.float32, device=device) for ag in agent_names}
            next_done = {ag: torch.zeros((), dtype=torch.float32, device=device) for ag in agent_names}

    # Store episode stats
    for ag in agent_names:
        episode_returns[ag].extend(ep_returns_batch[ag])
        episode_lengths[ag].extend(ep_lens_batch[ag])
        episode_hits[ag].extend(ep_hits_batch[ag])

    # Bootstrap values for GAE
    with torch.no_grad():
        next_values = {ag: nets[ag].get_value(next_obs[ag].unsqueeze(0)).squeeze(0) for ag in agent_names}

    # Compute advantages and returns using GAE
    for ag in agent_names:
        lastgaelam = 0.0
        for t in reversed(range(steps_per_update)):
            if t == steps_per_update - 1:
                next_nonterminal = 1.0 - float(next_done[ag].item())
                next_value = next_values[ag]
            else:
                next_nonterminal = 1.0 - float(buf[ag]["dones"][t + 1].item())
                next_value = buf[ag]["vals"][t + 1]
            delta = buf[ag]["rews"][t] + gamma * next_value * next_nonterminal - buf[ag]["vals"][t]
            lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
            buf[ag]["advantages"][t] = lastgaelam
        buf[ag]["returns"] = buf[ag]["advantages"] + buf[ag]["vals"]

    # POLICY UPDATE
    bsz = steps_per_update
    minibatch_size = bsz // num_minibatches

    metrics = {}
    for ag in agent_names:
        # Normalize advantages
        adv = buf[ag]["advantages"]
        adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf[ag]["advantages"] = adv_normalized

        # Collect metrics for this agent
        policy_losses = []
        value_losses = []
        entropy_losses = []
        ratios = []
        
        inds = np.arange(bsz)
        for epoch in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, bsz, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                obs_b = buf[ag]["obs"][mb_inds]
                acts_b = buf[ag]["acts"][mb_inds]
                logp_old_b = buf[ag]["logp"][mb_inds]
                adv_b = buf[ag]["advantages"][mb_inds]
                ret_b = buf[ag]["returns"][mb_inds]
                val_b = buf[ag]["vals"][mb_inds]

                _, logp, entropy, value = nets[ag].get_action_and_value(obs_b, acts_b)
                
                # Policy loss with ratio clipping
                ratio = (logp - logp_old_b).exp()
                ratios.extend(ratio.detach().cpu().numpy())
                
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping
                v_loss_unclipped = (value - ret_b) ** 2
                v_clipped = val_b + torch.clamp(value - val_b, -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - ret_b) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                # Gradient step
                opts[ag].zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(nets[ag].parameters(), max_grad_norm)
                opts[ag].step()
                
                # Store metrics
                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Update learning rate
        schedulers[ag].step()

        # Compute explained variance
        with torch.no_grad():
            v_pred = safecpu(nets[ag].get_value(buf[ag]["obs"]).cpu())
            v_true = safecpu(buf[ag]["returns"].cpu())
            ev = explained_variance(v_pred, v_true)
            
            metrics[ag] = {
                "ev": ev,
                "policy_loss": np.mean(policy_losses),
                "value_loss": np.mean(value_losses),
                "entropy": np.mean(entropy_losses),
                "ratio_mean": np.mean(ratios),
                "ratio_std": np.std(ratios),
                "lr": schedulers[ag].get_last_lr()[0]
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
                'optimizer_state_dict': opts[ag].state_dict(),
                'scheduler_state_dict': schedulers[ag].state_dict(),
                'update': update,
                'global_step': global_step,
                'metrics': metrics[ag]
            }, path)
        
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
        'optimizer_state_dict': opts[ag].state_dict(),
        'scheduler_state_dict': schedulers[ag].state_dict(),
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