import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_PPO import PointMassEnv
from ppo_network import PPO

def compute_gae(rewards, values, last_value, gamma=0.99, lam=0.95):
    """
    Compute generalized advantage estimation (GAE).
    Args:
        rewards:   np.array, shape [T]
        values:    np.array, shape [T]
        last_value: float, bootstrap value for last state
        gamma:     discount factor
        lam:       GAE lambda
    Returns:
        advantages: np.array, shape [T]
        returns:    np.array, shape [T]
    """
    T = len(rewards)
    values = np.append(values, last_value)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_update(policy, optimizer, obs_buf, act_buf, logp_old_buf, ret_buf, adv_buf,
               clip_eps=0.2, update_epochs=10, batch_size=64):
    """
    Perform PPO-clip update.
    """
    # normalize advantages
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
    data_size = len(obs_buf)
    for _ in range(update_epochs):
        idxs = np.random.permutation(data_size)
        for start in range(0, data_size, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]

            obs_batch = torch.as_tensor(obs_buf[batch_idx], dtype=torch.float32)
            act_batch = torch.as_tensor(act_buf[batch_idx], dtype=torch.float32)
            old_logp_batch = torch.as_tensor(logp_old_buf[batch_idx], dtype=torch.float32)
            ret_batch = torch.as_tensor(ret_buf[batch_idx], dtype=torch.float32)
            adv_batch = torch.as_tensor(adv_buf[batch_idx], dtype=torch.float32)

            dist, value = policy(obs_batch)
            logp = dist.log_prob(act_batch)
            ratio = torch.exp(logp - old_logp_batch)

            # policy loss
            clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_batch
            loss_pi = -(torch.min(ratio * adv_batch, clip_adv)).mean()

            # value loss
            loss_v = F.mse_loss(value, ret_batch)

            # entropy bonus
            entropy = dist.entropy().mean()

            # total loss
            loss = loss_pi + 0.5 * loss_v - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()


# --- Training parameters and initialization ---
steps = 500
epochs = 500
gamma = 0.99
lam = 0.95
clip_eps = 0.2
update_epochs = 10
batch_size = 64
lr = 3e-4

# Prepare environment and policy
env = PointMassEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

policy = PPO(obs_dim=obs_dim, act_dim=act_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

# Directories for saving
os.makedirs('models', exist_ok=True)

# Storage buffers
obs_buf = np.zeros((steps, obs_dim), dtype=np.float32)
act_buf = np.zeros((steps, act_dim), dtype=np.float32)
logp_buf = np.zeros(steps, dtype=np.float32)
val_buf = np.zeros(steps, dtype=np.float32)
rew_buf = np.zeros(steps, dtype=np.float32)

global_step = 0
episode_count = 0
# Training loop
for epoch in range(1, epochs + 1):
    o, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    # Collect experience
    for t in range(steps):
        obs_buf[t] = o
        with torch.no_grad():
            a, logp, v = policy.get_action(torch.as_tensor(o, dtype=torch.float32))
        a = a.numpy()
        logp_buf[t] = logp.numpy()
        val_buf[t] = v.numpy()
        act_buf[t] = a

        o2, r, terminated, truncated, info = env.step(a)
        rew_buf[t] = r
        ep_ret += r
        ep_len += 1
        global_step += 1

        o = o2
        timeout = ep_len >= env.max_steps
        if terminated or truncated or timeout:
            # Print per-episode log as in train_sb3_ppo_moving
            episode_count += 1
            distance = info.get('distance', np.nan)
            print(f"Episode {episode_count} | Reward: {ep_ret:.2f} | Distance: {distance:.2f}")
            # Bootstrap value
            if timeout or truncated:
                with torch.no_grad():
                    _, _, v = policy.get_action(torch.as_tensor(o, dtype=torch.float32))
                last_val = v.numpy()
            else:
                last_val = 0.0
            # Reset episode
            o, _ = env.reset()
            ep_ret, ep_len = 0.0, 0
    # Compute GAE and update
    adv_buf, ret_buf = compute_gae(rew_buf, val_buf, last_val, gamma, lam)
    ppo_update(policy, optimizer, obs_buf, act_buf, logp_buf, ret_buf, adv_buf,
               clip_eps, update_epochs, batch_size)

# Save final model
ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
model_path = f'models/ppo_custom_moving_target_{ts}.zip'
torch.save(policy.state_dict(), model_path)
print("\nTraining completed!")