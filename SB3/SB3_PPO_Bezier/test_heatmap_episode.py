import numpy as np
from env_sb3_ppo_heatmap import PointMassEnv
import math

env = PointMassEnv()
obs, _ = env.reset()
step = 1
prev_target_dir = np.array([0.0, 0.0, 0.0])

while True:
    x, y, z, v, yaw, pitch = env.state
    agent_pos = np.array([x, y, z])
    target_pos = env.target_state.copy()

    dist_air = np.linalg.norm(agent_pos - target_pos)

    agent_dir = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch)
    ])

    # Calcolo la direzione del target basandomi sullo storico
    if len(env.target_history) > 1:
        cur_t = env.target_history[-1]
        prev_t = env.target_history[-2]
        delta_t = cur_t - prev_t
        norm_dt = np.linalg.norm(delta_t) + 1e-8
        target_dir = delta_t / norm_dt
    else:
        target_dir = prev_target_dir

    prev_target_dir = target_dir.copy()

    # Calcola l’angolo fra i due vettori
    cos_ang = np.clip(np.dot(agent_dir, target_dir), -1.0, 1.0)
    ang_diff = math.degrees(math.acos(cos_ang))

    # Desired yaw e pitch per inseguire il target
    desired_yaw = math.atan2(target_dir[1], target_dir[0])
    desired_pitch = math.asin(np.clip(target_dir[2], -1.0, 1.0))

    # 1) Calcolo delta yaw: differenza normalizzata in [-pi, +pi]
    raw_yaw_diff = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
    max_dyaw = env.delta_yaw
    dyaw = float(np.clip(raw_yaw_diff, -max_dyaw, max_dyaw))

    # 2) Calcolo delta pitch: differenza normalizzata in [-pi/2, +pi/2]
    raw_pitch_diff = desired_pitch - pitch
    max_dpitch = env.delta_pitch
    dpitch = float(np.clip(raw_pitch_diff, -max_dpitch, max_dpitch))

    # 3) Calcolo delta v: vogliamo un valore di velocità proporzionale alla distanza
    d = np.linalg.norm(target_pos - agent_pos) + 1e-8
    if d > 2.0:
        target_speed = env.v_max
    else:
        target_speed = env.v_min
    raw_dv = target_speed - v
    max_dv = env.delta_v
    dv = float(np.clip(raw_dv, -max_dv, max_dv))

    # Assign the action
    action = np.array([dv, dyaw, dpitch], dtype=np.float32)

    # Execute a step
    obs, reward, done, truncated, info = env.step(action)

    # Useful info for logging
    ax, ay, az = agent_pos
    tx, ty, tz = target_pos
    adx, ady, adz = agent_dir
    tdx, tdy, tdz = target_dir

    print(
        f"Step {step:03d} | "
        f"Ag: [{ax:.2f}, {ay:.2f}, {az:.2f}] | "
        f"Tar: [{tx:.2f}, {ty:.2f}, {tz:.2f}] | "
        f"Dist: {dist_air:.2f} | "
        f"DirAg: [{adx:.2f}, {ady:.2f}, {adz:.2f}] | "
        f"DirTar: [{tdx:.2f}, {tdy:.2f}, {tdz:.2f}] | "
        f"ΔAng: {ang_diff:.1f}° | "
        f"Rew: {reward:.4f}"
    )

    step += 1

    if truncated:
        break