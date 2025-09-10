import os
import sys
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from matplotlib.animation import FuncAnimation, PillowWriter

# ensure relative imports work when running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.tag_shoot_env import TagShootEnv
from algorithms.IPPO.models import EnhancedActorCritic

# Helper functions
def find_latest_ckpt(base_dir="policies"):
    fixed_dirs = glob.glob(os.path.join(base_dir, "Tag_Shoot_IPPO", "*"))
    if fixed_dirs:
        latest = max(fixed_dirs, key=os.path.getmtime)
        print(f"Found training run: {latest}")
    else:
        # Fallback to original directory structure
        runs = glob.glob(os.path.join(base_dir, "Tag_Shoot_IPPO", "*"))
        if not runs:
            raise FileNotFoundError(f"No runs found in {base_dir}")
        latest = max(runs, key=os.path.getmtime)
    
    # Try per-agent checkpoints
    ckpts = glob.glob(os.path.join(latest, "*_policy_*.pth"))
    ch, ev = None, None
    
    if ckpts:
        chs = [p for p in ckpts if "Agent 0" in os.path.basename(p)]
        evs = [p for p in ckpts if "Agent 1" in os.path.basename(p)]
        if chs and evs:
            # Get the latest checkpoint for each agent
            ch = max(chs, key=lambda x: int(x.split('upd')[1].split('.pth')[0]) if 'upd' in x else 0)
            ev = max(evs, key=lambda x: int(x.split('upd')[1].split('.pth')[0]) if 'upd' in x else 0)
            print(f"Using Agent 0 checkpoint: {os.path.basename(ch)}")
            print(f"Using Agent 1 checkpoint: {os.path.basename(ev)}")
    
    # Fallback: single shared policy
    if ch is None or ev is None:
        shared = glob.glob(os.path.join(latest, "*shared*_*.pth")) or glob.glob(os.path.join(latest, "*policy_*.pth"))
        if not shared:
            raise FileNotFoundError(f"No checkpoints found in {latest}")
        shared = max(shared, key=os.path.getmtime)
        ch = shared
        ev = shared
        print(f"Using shared checkpoint: {os.path.basename(shared)}")
    
    return ch, ev, latest

def policy_from_ckpt(path, obs_dim, act_dim, device="cpu"):
    """Load policy from checkpoint"""
    net = EnhancedActorCritic(obs_dim, act_dim)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    net.load_state_dict(state_dict)
    print(f"Loaded model from {os.path.basename(path)}")
    
    net.eval()
    return net.to(device)

def _nose_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cp = math.cos(pitch)
    return np.array([cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)], dtype=np.float64)

def _step_engagement_features(env, prev_d: float | None, prox_factor: float = 2.0):
    s0 = env.states['Agent 0']; s1 = env.states['Agent 1']
    p0, p1 = s0[:3], s1[:3]
    y0, ph0 = float(s0[4]), float(s0[5])
    y1, ph1 = float(s1[4]), float(s1[5])

    v0 = _nose_from_yaw_pitch(y0, ph0)
    v1 = _nose_from_yaw_pitch(y1, ph1)

    r01 = p1 - p0
    d = float(np.linalg.norm(r01))
    if d < 1e-8:
        los01 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        los01 = r01 / d
    los10 = -los01

    # Puntamento ∈ [0,1] per ciascun agente, poi media
    f0 = 0.5 * (1.0 + float(np.clip(np.dot(v0, los01), -1.0, 1.0)))
    f1 = 0.5 * (1.0 + float(np.clip(np.dot(v1, los10), -1.0, 1.0)))
    f_point_avg = 0.5 * (f0 + f1)

    # Prossimità (entro 2× WEZ per default)
    in_prox = 1.0 if d <= prox_factor * float(env.wez_length) else 0.0

    # Presenza in WEZ (almeno uno dei due)
    try:
        in_wez_any = 1.0 if (env._in_wez('Agent 0') or env._in_wez('Agent 1')) else 0.0
    except Exception:
        # fallback conservativo se la funzione non è esposta
        in_wez_any = 1.0 if (in_prox and (f0 > 0.5 or f1 > 0.5)) else 0.0

    # Chiusura (diminuzione distanza fra step)
    is_closing = 0.0
    if prev_d is not None and d < prev_d - 1e-6:
        is_closing = 1.0

    return f_point_avg, in_prox, in_wez_any, is_closing, d

# ==== Evaluation helpers (WEZ, closure, orbita, PMD, lock-gating) ====
def _speed_vec_from_state(s: np.ndarray) -> np.ndarray:
    v, yaw, pitch = float(s[3]), float(s[4]), float(s[5])
    cp = math.cos(pitch)
    return np.array([v * cp * math.cos(yaw), v * cp * math.sin(yaw), v * math.sin(pitch)], dtype=np.float64)

def _los_and_d(p_i: np.ndarray, p_j: np.ndarray):
    r = p_j - p_i
    d = float(np.linalg.norm(r))
    if d < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    return (r / d), d

def _closure_from_states(s_i: np.ndarray, s_j: np.ndarray) -> float:
    p_i, p_j = s_i[:3].astype(np.float64), s_j[:3].astype(np.float64)
    los, _ = _los_and_d(p_i, p_j)
    v_rel = _speed_vec_from_state(s_i) - _speed_vec_from_state(s_j)
    return -float(np.dot(v_rel, los))  # >0 se chiudi

def _relative_components_eval(s_i: np.ndarray, s_j: np.ndarray):
    """vr (>0 chiusura), vt (tangenziale), cos_orbit (co-planarità)"""
    p_i, p_j = s_i[:3].astype(np.float64), s_j[:3].astype(np.float64)
    v_i, v_j = _speed_vec_from_state(s_i), _speed_vec_from_state(s_j)
    los, d = _los_and_d(p_i, p_j)
    v_rel = v_i - v_j
    vr = -float(np.dot(v_rel, los))
    vt = float(np.linalg.norm(np.cross(v_rel, los)))
    h_i = np.cross(p_j - p_i, v_i); h_j = np.cross(p_j - p_i, v_j)
    nh_i, nh_j = np.linalg.norm(h_i), np.linalg.norm(h_j)
    cos_orbit = 0.0 if nh_i < 1e-9 or nh_j < 1e-9 else float(np.clip(np.dot(h_i, h_j) / (nh_i * nh_j), -1.0, 1.0))
    return vr, vt, cos_orbit, d

def _los_rate_and_pmd_eval(s_i: np.ndarray, s_j: np.ndarray):
    p_i, p_j = s_i[:3].astype(np.float64), s_j[:3].astype(np.float64)
    v_i, v_j = _speed_vec_from_state(s_i), _speed_vec_from_state(s_j)
    r = p_j - p_i
    d2 = float(np.dot(r, r)) + 1e-12
    v_rel = v_j - v_i
    los_rate = float(np.linalg.norm(np.cross(v_rel, r)) / d2)
    vrel_n = float(np.linalg.norm(v_rel)) + 1e-12
    pmd = float(np.linalg.norm(np.cross(r, v_rel)) / vrel_n)
    return los_rate, pmd

def _in_wez_eval(env: TagShootEnv, shooter: np.ndarray, target: np.ndarray) -> bool:
    rel = target[:3] - shooter[:3]
    d = float(np.linalg.norm(rel))
    if d < 1e-8:
        return False
    los = rel / d
    cp = math.cos(float(shooter[5]))
    nose = np.array([cp * math.cos(float(shooter[4])), cp * math.sin(float(shooter[4])), math.sin(float(shooter[5]))])
    cos_nose = float(np.dot(nose, los))
    return (d <= float(env.wez_length)) and (cos_nose >= float(env.wez_cos_threshold))

def _good_lock_geom_eval(env: TagShootEnv, shooter: np.ndarray, target: np.ndarray) -> bool:
    """Replica il lock-gating: WEZ + closure>0 + nose·LOS/LEAD buoni + PMD bassa + cooldown=0."""
    # cooldown check (env espone i dict)
    shooter_name = 'Agent 0' if np.all(shooter is env.states['Agent 0']) else 'Agent 1'
    if env.cooldown.get(shooter_name, 0) > 0:
        return False

    p_i, p_j = shooter[:3], target[:3]
    los, d = _los_and_d(p_i, p_j)
    # nose shooter
    cp = math.cos(float(shooter[5]))
    nose = np.array([cp * math.cos(float(shooter[4])), cp * math.sin(float(shooter[4])), math.sin(float(shooter[5]))])
    # lead (grezzo come in env)
    vhat_j = _speed_vec_from_state(target); vj = float(np.linalg.norm(vhat_j)) + 1e-6
    t_lead = d / max(1e-6, float(env.v_max))
    lead_dir = p_j + (vhat_j / vj) * float(env.v_max) * t_lead - p_i
    n = np.linalg.norm(lead_dir); lead_u = (lead_dir / n) if n > 1e-8 else los

    closure = _closure_from_states(shooter, target)
    _, pmd = _los_rate_and_pmd_eval(shooter, target)
    in_wez = _in_wez_eval(env, shooter, target)

    cos_th_wez = float(env.wez_cos_threshold)
    cos_th_lead = float(math.cos(math.radians(22.0)))  # tieni allineato all'env
    return (in_wez and closure > 0.0
            and float(np.dot(nose, los)) >= cos_th_wez
            and float(np.dot(nose, lead_u)) >= cos_th_lead
            and pmd <= 0.35 * float(env.wez_length))

# Plotting helper functions
def set_axes_cube(ax, center, radius):
    """Set 3D axes limits to create a cube around the center point."""
    cx, cy, cz = center
    r = float(radius)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    
def plot_traj_with_hits(ch, ev, hits_on_0, hits_on_1, title, outdir, episode_info=None):
    """
    hits_on_0 / hits_on_1: list of indices (frame numbers) where that agent got hit
    """
    fig = plt.figure(figsize=(8, 7), dpi=130)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(ch[:, 0], ch[:, 1], ch[:, 2], label="Agent 0", linewidth=2)
    ax.plot(ev[:, 0], ev[:, 1], ev[:, 2], "--", label="Agent 1", linewidth=2)

    ax.scatter(ch[0, 0], ch[0, 1], ch[0, 2], s=80, marker="o", label="A0 start", alpha=0.8)
    ax.scatter(ev[0, 0], ev[0, 1], ev[0, 2], s=80, marker="o", label="A1 start", alpha=0.8)
    ax.scatter(ch[-1, 0], ch[-1, 1], ch[-1, 2], s=90, marker="^", label="A0 end", alpha=0.8)
    ax.scatter(ev[-1, 0], ev[-1, 1], ev[-1, 2], s=90, marker="^", label="A1 end", alpha=0.8)

    # X markers where hits occurred
    def clamp_idx(i, N): return max(0, min(int(i), N - 1))
    if hits_on_0:
        idxs = [clamp_idx(i, len(ch)) for i in hits_on_0]
        ax.scatter(ch[idxs, 0], ch[idxs, 1], ch[idxs, 2], marker="x", s=120, linewidths=3, 
                  color='red', label="Hit on A0", alpha=0.9)
    if hits_on_1:
        idxs = [clamp_idx(i, len(ev)) for i in hits_on_1]
        ax.scatter(ev[idxs, 0], ev[idxs, 1], ev[idxs, 2], marker="x", s=120, linewidths=3, 
                  color='red', label="Hit on A1", alpha=0.9)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # Enhanced title with episode info
    if episode_info:
        title_text = f"{title}\n{episode_info}"
    else:
        title_text = title
    ax.set_title(title_text, fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # fit cube tight on both trajs
    data = np.vstack([ch, ev])
    mins, maxs = data.min(0), data.max(0)
    center = 0.5 * (mins + maxs)
    radius = max((maxs - mins) / 2.0)
    radius = float(max(radius, 2.0))  # Minimum radius for better visibility
    set_axes_cube(ax, center, radius)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{title}.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"Saved {path}")

def animate_traj_follow_zoom(ch, ev, hits_on_0, hits_on_1, title, outdir,
                             window=60, margin=2.0, min_radius=4.0, fps=20, dpi=120):
    """
    Enhanced GIF animation with better following and hit visualization
    """
    T = min(len(ch), len(ev))
    ch = np.asarray(ch[:T], dtype=np.float64)
    ev = np.asarray(ev[:T], dtype=np.float64)

    hits_on_0 = [i for i in hits_on_0 if 0 <= i < T]
    hits_on_1 = [i for i in hits_on_1 if 0 <= i < T]

    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=12)

    # lines and points
    a0_line, = ax.plot([], [], [], lw=2.5, label="Agent 0", alpha=0.8)
    a1_line, = ax.plot([], [], [], lw=2.5, linestyle="--", label="Agent 1", alpha=0.8)
    a0_pt, = ax.plot([], [], [], marker="o", markersize=8, linestyle="None", color='blue')
    a1_pt, = ax.plot([], [], [], marker="o", markersize=8, linestyle="None", color='orange')

    # Hit markers
    hit_a0_scat = ax.scatter([], [], [], marker="x", s=120, linewidths=3, color='red', alpha=0.9)
    hit_a1_scat = ax.scatter([], [], [], marker="x", s=120, linewidths=3, color='red', alpha=0.9)

    ax.legend(loc="upper right")

    # smooth camera tracking
    center_ema = None
    radius_ema = None
    alpha = 0.25  # Camera smoothing

    def update(i):
        nonlocal center_ema, radius_ema
        i = int(i)
        start = max(0, i - window + 1)

        # Update trajectory lines
        a0_hist = ch[:i + 1]
        a1_hist = ev[:i + 1]
        a0_line.set_data(a0_hist[:, 0], a0_hist[:, 1])
        a0_line.set_3d_properties(a0_hist[:, 2])
        a1_line.set_data(a1_hist[:, 0], a1_hist[:, 1])
        a1_line.set_3d_properties(a1_hist[:, 2])

        # Update current position markers
        a0_pt.set_data([ch[i, 0]], [ch[i, 1]]); a0_pt.set_3d_properties([ch[i, 2]])
        a1_pt.set_data([ev[i, 0]], [ev[i, 1]]); a1_pt.set_3d_properties([ev[i, 2]])

        # Update hit markers (only hits that have occurred so far)
        a0_idxs = [j for j in hits_on_0 if j <= i]
        a1_idxs = [j for j in hits_on_1 if j <= i]
        
        if a0_idxs:
            a0p = ch[a0_idxs]
            hit_a0_scat._offsets3d = (a0p[:, 0], a0p[:, 1], a0p[:, 2])
        else:
            hit_a0_scat._offsets3d = ([], [], [])
            
        if a1_idxs:
            a1p = ev[a1_idxs]
            hit_a1_scat._offsets3d = (a1p[:, 0], a1p[:, 1], a1p[:, 2])
        else:
            hit_a1_scat._offsets3d = ([], [], [])

        # Dynamic camera: focus on the action
        current_center = 0.5 * (ch[i] + ev[i])
        
        # Consider recent trajectory for bounding
        w0 = ch[start:i + 1]
        w1 = ev[start:i + 1]
        data = np.vstack([w0, w1])
        mins, maxs = data.min(0), data.max(0)
        half_ranges = 0.5 * (maxs - mins)
        r_local = float(np.max(half_ranges) + margin)
        
        # Also consider current separation
        r_pair = float(np.linalg.norm(ch[i] - ev[i]) * 0.4 + margin)
        r = max(r_local, r_pair, min_radius)

        # Smooth camera movement
        if center_ema is None:
            center_ema = current_center
            radius_ema = r
        else:
            center_ema = (1 - alpha) * center_ema + alpha * current_center
            radius_ema = (1 - alpha) * radius_ema + alpha * r

        set_axes_cube(ax, center_ema, radius_ema)
        return a0_line, a1_line, a0_pt, a1_pt

    ani = FuncAnimation(fig, update, frames=T, interval=int(1000 / fps), blit=False)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{title}.gif")
    try:
        ani.save(path, writer=PillowWriter(fps=fps))
        print(f"Saved {path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    finally:
        plt.close()

# Evaluation run
ch_path, ev_path, run_dir = find_latest_ckpt()
env = TagShootEnv()

obs_dim = env.observation_spaces['Agent 0'].shape[0]
act_dim = env.action_spaces['Agent 0'].shape[0]
dev = "cpu"

print(f"Environment obs_dim: {obs_dim}, act_dim: {act_dim}")

# Load policies with fallback
ch_net = policy_from_ckpt(ch_path, obs_dim, act_dim, device=dev)
ev_net = policy_from_ckpt(ev_path, obs_dim, act_dim, device=dev)

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
outdir = os.path.join("plots", "Tag_Shoot_IPPO", f"eval_{ts}")
os.makedirs(outdir, exist_ok=True)

num_eps = 50
best_metric = -np.inf
best_idx = 0
episodes = []  # (A0_traj, A1_traj, hits_on_0, hits_on_1, metric, episode_stats)

print(f"Running {num_eps} evaluation episodes...")

for ep in range(num_eps):
    obs, infos = env.reset()
    point_sum = 0.0
    # --- extra accumulators (good-lock, orbita, divergenza, PMD) ---
    good_lock_steps = 0
    lock_steps = 0
    orbit_score_sum = 0.0
    diverge_steps = 0
    pmd_in_wez = []

    # utility per normalizzazioni
    vmax = float(env.v_max)
    wezL = float(env.wez_length)
    losrate_ref = float(vmax / max(1e-6, wezL))

    prox_count = 0.0
    wez_count = 0.0
    closing_count = 0.0
    steps_metric = 0
    prev_d_for_metric = None
    done = False
    a0_traj, a1_traj = [], []
    hp0_hist, hp1_hist = [], []

    # Initial HP snapshot
    hp0_hist.append(int(env.hp['Agent 0']))
    hp1_hist.append(int(env.hp['Agent 1']))

    step_count = 0
    while not done and step_count < 2000:  # Safety limit
        a0_traj.append(env.states['Agent 0'][:3].copy())
        a1_traj.append(env.states['Agent 1'][:3].copy())

        actions = {}
        with torch.no_grad():
            for name, net in [('Agent 0', ch_net), ('Agent 1', ev_net)]:
                o = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
                a, _, _, _ = net.get_action_and_value(o)
                low = torch.as_tensor(env.action_spaces[name].low, dtype=torch.float32)
                high = torch.as_tensor(env.action_spaces[name].high, dtype=torch.float32)
                a = torch.clamp(a.squeeze(0), low, high)
                actions[name] = a.cpu().numpy()

        obs, rews, dones, infos = env.step(actions)
        hp0_hist.append(int(env.hp['Agent 0']))
        hp1_hist.append(int(env.hp['Agent 1']))
        fpt, in_prox, in_wez_any, is_closing, d_now = _step_engagement_features(env, prev_d_for_metric)
        # stati aggiornati
        s0 = env.states['Agent 0']; s1 = env.states['Agent 1']

        # good-lock per ciascun lato (coerente col gating dell'env)
        is_lock0 = _in_wez_eval(env, s0, s1) and env.cooldown['Agent 0'] == 0
        is_lock1 = _in_wez_eval(env, s1, s0) and env.cooldown['Agent 1'] == 0
        lock_steps += int(_in_wez_eval(env, s0, s1)) + int(_in_wez_eval(env, s1, s0))

        good0 = _good_lock_geom_eval(env, s0, s1)
        good1 = _good_lock_geom_eval(env, s1, s0)
        good_lock_steps += int(_good_lock_geom_eval(env, s0, s1)) + int(_good_lock_geom_eval(env, s1, s0))

        # orbita: vt alto + co-planarità alta (solo quando c'è ingaggio vicino)
        vr, vt, cos_orb, _ = _relative_components_eval(s0, s1)
        vt_n = float(np.clip(vt / max(1e-6, vmax), 0.0, 1.0))
        orbit_score = 0.5 * vt_n + 0.5 * (0.5 * (1.0 + float(cos_orb)))  # 0..1
        if in_wez_any > 0.0:
            orbit_score_sum += orbit_score

        # divergenza “allineato ma scappo”: penalizza comportamenti brutti
        # (valuta lato A0 -> A1; simmetria equivalente)
        cp = _nose_from_yaw_pitch(float(s0[4]), float(s0[5]))
        los01, _ = _los_and_d(s0[:3], s1[:3])
        cos_point = float(np.clip(np.dot(cp, los01), -1.0, 1.0))
        if (cos_point > 0.85) and (_closure_from_states(s0, s1) < -0.02) and (d_now > 0.6 * wezL):
            diverge_steps += 1

        # PMD medio quando almeno un lato è in WEZ
        if _in_wez_eval(env, s0, s1) or _in_wez_eval(env, s1, s0):
            _, pmd01 = _los_rate_and_pmd_eval(s0, s1)
            pmd_in_wez.append(pmd01)

        point_sum += fpt
        prox_count += in_prox
        wez_count  += in_wez_any
        closing_count += is_closing
        steps_metric += 1
        prev_d_for_metric = d_now

        done = any(dones.values())
        step_count += 1

    a0_traj = np.array(a0_traj, dtype=np.float64)
    a1_traj = np.array(a1_traj, dtype=np.float64)
    hp0_hist = np.array(hp0_hist, dtype=np.int32)
    hp1_hist = np.array(hp1_hist, dtype=np.int32)

    # ---- HITS (come prima) ----
    hits_on_0 = np.where(hp0_hist[1:] < hp0_hist[:-1])[0].tolist()
    hits_on_1 = np.where(hp1_hist[1:] < hp1_hist[:-1])[0].tolist()
    total_hits = len(hits_on_0) + len(hits_on_1)

    # ---- LOCKS “BUONI” (calcolati sopra) ----
    total_good_locks = int(good_lock_steps)
    lock_steps_total = int(lock_steps)
    good_lock_density = float(total_good_locks) / max(1, steps_metric)
    good_lock_density = float(np.clip(good_lock_density, 0.0, 1.0))
    conv_denominator = max(1, total_good_locks // max(1, int(env.lock_required)))
    lock_to_hit_conv = float(total_hits) / float(conv_denominator)

    # ---- ORBITA / DIVERGENZA / PMD ----
    orbit_score_avg = orbit_score_sum / max(1, int(wez_count))  # orbit solo quando in WEZ
    diverge_share   = float(diverge_steps) / max(1, steps_metric)
    pmd_mean_in_wez = float(np.mean(pmd_in_wez)) if len(pmd_in_wez) > 0 else float('nan')

    # ---- Metrica engagement rivista ----
    point_avg   = point_sum / max(1, steps_metric)
    prox_share  = prox_count / max(1, steps_metric)
    wez_share   = wez_count  / max(1, steps_metric)
    close_share = closing_count / max(1, steps_metric - 1)

    engagement_metric = (
        12.0 * total_hits           # impatto reale
    + 3.0  * lock_to_hit_conv     # conversione lock→hit
    + 2.0  * orbit_score_avg      # orbita stretta / co-planare
    + 2.0  * wez_share            # tempo in WEZ
    + 1.5  * prox_share           # vicinanza
    + 1.5  * close_share          # chiudono
    + 0.8  * good_lock_density    # lock di qualità, non “farm”
    - 2.0  * diverge_share        # penalizza "allineato ma scappo"
    )

    episode_stats = {
        # principali
        'total_hits': total_hits,
        'final_hp': (int(env.hp['Agent 0']), int(env.hp['Agent 1'])),
        'episode_length': len(a0_traj),
        'engagement_metric': engagement_metric,
        # breakdown vecchio+nuovo
        'point_avg': point_avg,
        'prox_share': prox_share,
        'wez_share': wez_share,
        'close_share': close_share,
        # qualità lock e conversione
        'good_locks': total_good_locks,
        'lock_steps_total': lock_steps_total,
        'good_lock_density': good_lock_density,
        'lock_to_hit_conv': lock_to_hit_conv,
        # orbita/divergenza/PMD
        'orbit_score_avg': orbit_score_avg,
        'diverge_share': diverge_share,
        'pmd_mean_in_wez': pmd_mean_in_wez,
    }


    episodes.append((a0_traj, a1_traj, hits_on_0, hits_on_1, engagement_metric, episode_stats))

    if engagement_metric > best_metric:
        best_metric = engagement_metric
        best_idx = ep

    if (ep + 1) % 10 == 0:
        print(f"Completed {ep + 1}/{num_eps} episodes")

print(f"\nEvaluation completed!")
print(f"Best episode #{best_idx + 1} with engagement metric={best_metric:.2f}")

# Plot and animate best episode
best = episodes[best_idx]
best_info  = f"GoodLocks: {best[5]['good_locks']} | Hits: {best[5]['total_hits']} | Conv: {best[5]['lock_to_hit_conv']:.2f} | HP: {best[5]['final_hp']}"
plot_traj_with_hits(best[0], best[1], best[2], best[3], 
                    f"best_episode_{best_idx + 1}", outdir, best_info)
animate_traj_follow_zoom(best[0], best[1], best[2], best[3], 
                        f"best_episode_{best_idx + 1}_follow", outdir)

# Plot and animate worst episode
worst_idx = int(np.argmin([e[4] for e in episodes]))
worst = episodes[worst_idx]
worst_info = f"GoodLocks: {worst[5]['good_locks']} | Hits: {worst[5]['total_hits']} | Conv: {worst[5]['lock_to_hit_conv']:.2f} | HP: {worst[5]['final_hp']}"
plot_traj_with_hits(worst[0], worst[1], worst[2], worst[3], 
                    f"worst_episode_{worst_idx + 1}", outdir, worst_info)
animate_traj_follow_zoom(worst[0], worst[1], worst[2], worst[3], 
                        f"worst_episode_{worst_idx + 1}_follow", outdir)

# Print summary statistics
print(f"\n=== EVALUATION SUMMARY ===")
print(f"Total episodes: {num_eps}")
total_hits = sum(len(e[2]) + len(e[3]) for e in episodes)
total_locks = sum(e[5]['good_locks'] for e in episodes)
avg_length = np.mean([e[5]['episode_length'] for e in episodes])
    
print(f"Total hits across all episodes: {total_hits}")
print(f"Total locks across all episodes: {total_locks}")
print(f"Average episode length: {avg_length:.1f} steps")
print(f"Outputs saved to {outdir}")
print(f"Models loaded from {run_dir}")