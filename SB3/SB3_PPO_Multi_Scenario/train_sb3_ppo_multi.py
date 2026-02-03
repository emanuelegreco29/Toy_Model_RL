import os
import json
import datetime
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env_sb3_ppo_multi import PointMassEnv

ALGO = "SAC"  # "SAC"  # "TD3"

TOTAL_TIMESTEPS = 500 * 2000
SMOOTH_WINDOW = 25
K_HISTORY = 1
DEVICE = "cuda"


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size == 0:
        return x.copy()
    if x.size < window:
        return np.array([], dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="valid")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv_two_cols(path: str, timesteps: np.ndarray, values: np.ndarray, y_name: str) -> None:
    timesteps = np.asarray(timesteps, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"timestep,{y_name}\n")
        for t, v in zip(timesteps, values):
            if np.isnan(v):
                continue
            f.write(f"{int(t)},{float(v)}\n")


def plot_raw_and_smooth(
    out_png: str,
    out_pdf: str,
    timesteps_raw: np.ndarray,
    values_raw: np.ndarray,
    timesteps_smooth: np.ndarray,
    values_smooth: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)

    ax.plot(timesteps_raw, values_raw, alpha=0.2, linewidth=1.0, label="raw")
    if values_smooth.size > 0:
        ax.plot(timesteps_smooth, values_smooth, linewidth=2.0, alpha=0.9, label=f"smooth_{SMOOTH_WINDOW}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_components_smooth(
    out_png: str,
    out_pdf: str,
    timesteps_smooth: np.ndarray,
    comps_smooth: Dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111)

    for k, v in comps_smooth.items():
        if v.size == 0:
            continue
        ax.plot(timesteps_smooth, v, linewidth=2.0, alpha=0.9, label=k)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


@dataclass
class EpisodeRecord:
    timestep: int
    ep_reward: float
    final_distance: float
    totally_behind: float
    comp_f_dist: float
    comp_f_head_pos: float
    comp_f_head_vel: float
    comp_w_dist: float
    comp_w_head_pos: float
    comp_w_head_vel: float
    comp_bias: float
    comp_total_reward: float


class EpisodeLogger:
    def __init__(self):
        self.records: List[EpisodeRecord] = []

    def __call__(self, locals_, globals_):
        infos = locals_["infos"]
        model = locals_["self"]
        num_ts = int(getattr(model, "num_timesteps", 0))

        for info in infos:
            ep = info.get("episode")
            if not ep:
                continue

            ep_reward = float(ep.get("r", np.nan))
            final_distance = float(info.get("distance", np.nan))
            totally_behind = float(info.get("totally_behind", info.get("followed", np.nan)))

            rc = info.get("reward_components", {}) or {}
            comp_f_dist = float(rc.get("f_dist", np.nan))
            comp_f_head_pos = float(rc.get("f_head_pos", np.nan))
            comp_f_head_vel = float(rc.get("f_head_vel", np.nan))
            comp_w_dist = float(rc.get("w_dist", np.nan))
            comp_w_head_pos = float(rc.get("w_head_pos", np.nan))
            comp_w_head_vel = float(rc.get("w_head_vel", np.nan))
            comp_bias = float(rc.get("bias", np.nan))
            comp_total_reward = float(rc.get("total_reward", np.nan))

            self.records.append(
                EpisodeRecord(
                    timestep=num_ts,
                    ep_reward=ep_reward,
                    final_distance=final_distance,
                    totally_behind=totally_behind,
                    comp_f_dist=comp_f_dist,
                    comp_f_head_pos=comp_f_head_pos,
                    comp_f_head_vel=comp_f_head_vel,
                    comp_w_dist=comp_w_dist,
                    comp_w_head_pos=comp_w_head_pos,
                    comp_w_head_vel=comp_w_head_vel,
                    comp_bias=comp_bias,
                    comp_total_reward=comp_total_reward,
                )
            )

            n = len(self.records)
            print(
                f"Episode {n} | Timestep {num_ts} | Reward: {ep_reward:.2f} | "
                f"FinalDist: {final_distance:.2f} | TotallyBehind: {totally_behind:.0f}"
            )

        return True


env = DummyVecEnv([lambda: Monitor(PointMassEnv(K_history=K_HISTORY))])

policy_kwargs_ppo = dict(net_arch=[512, 512, 512], log_std_init=-1.0)
policy_kwargs_sac = dict(net_arch=[512, 512, 512], log_std_init=-1.0)
policy_kwargs_td3 = dict(net_arch=[512, 512, 512])

if ALGO == "PPO":
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs_ppo,
        learning_rate=1e-4,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=20,
        batch_size=512,
        verbose=0,
        device=DEVICE,
    )
elif ALGO == "SAC":
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs_sac,
        learning_rate=1e-4,
        batch_size=256,
        tau=0.005,
        verbose=0,
        device=DEVICE,
    )
elif ALGO == "TD3":
    model = TD3(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs_td3,
        learning_rate=1e-4,
        batch_size=256,
        tau=0.005,
        verbose=0,
        device=DEVICE,
    )
else:
    raise ValueError(f"Unknown ALGO={ALGO}")

callback = EpisodeLogger()
print(f"Starting training for: {ALGO}")
print("Using device:", model.device)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("models", f"{ALGO.lower()}_sb3_multi_{ts}")
csv_dir = os.path.join(run_dir, "csv")
fig_dir = os.path.join(run_dir, "figures")
ensure_dir(csv_dir)
ensure_dir(fig_dir)

model_path = os.path.join(run_dir, f"{ALGO.lower()}_model.zip")
model.save(model_path)

t = np.array([r.timestep for r in callback.records], dtype=np.int64)

train_reward = np.array([r.ep_reward for r in callback.records], dtype=float)
final_distance = np.array([r.final_distance for r in callback.records], dtype=float)
totally_behind = np.array([r.totally_behind for r in callback.records], dtype=float)

comp_f_dist = np.array([r.comp_f_dist for r in callback.records], dtype=float)
comp_f_head_pos = np.array([r.comp_f_head_pos for r in callback.records], dtype=float)
comp_f_head_vel = np.array([r.comp_f_head_vel for r in callback.records], dtype=float)
comp_w_dist = np.array([r.comp_w_dist for r in callback.records], dtype=float)
comp_w_head_pos = np.array([r.comp_w_head_pos for r in callback.records], dtype=float)
comp_w_head_vel = np.array([r.comp_w_head_vel for r in callback.records], dtype=float)
comp_bias = np.array([r.comp_bias for r in callback.records], dtype=float)
comp_total_reward = np.array([r.comp_total_reward for r in callback.records], dtype=float)

reward_s = moving_average(train_reward, SMOOTH_WINDOW)
dist_s = moving_average(final_distance, SMOOTH_WINDOW)
behind_s = moving_average(totally_behind, SMOOTH_WINDOW)

comp_f_dist_s = moving_average(comp_f_dist, SMOOTH_WINDOW)
comp_f_head_pos_s = moving_average(comp_f_head_pos, SMOOTH_WINDOW)
comp_f_head_vel_s = moving_average(comp_f_head_vel, SMOOTH_WINDOW)
comp_w_dist_s = moving_average(comp_w_dist, SMOOTH_WINDOW)
comp_w_head_pos_s = moving_average(comp_w_head_pos, SMOOTH_WINDOW)
comp_w_head_vel_s = moving_average(comp_w_head_vel, SMOOTH_WINDOW)
comp_bias_s = moving_average(comp_bias, SMOOTH_WINDOW)
comp_total_reward_s = moving_average(comp_total_reward, SMOOTH_WINDOW)

t_s = t[SMOOTH_WINDOW - 1 :] if t.size >= SMOOTH_WINDOW else np.array([], dtype=np.int64)

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_reward.csv"), t, train_reward, "reward")
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_reward_smooth_{SMOOTH_WINDOW}.csv"), t_s, reward_s, "reward_smooth")

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_final_distance.csv"), t, final_distance, "final_distance")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_final_distance_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    dist_s,
    "final_distance_smooth",
)

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_totally_behind.csv"), t, totally_behind, "totally_behind")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_totally_behind_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    behind_s,
    "totally_behind_smooth",
)

# Reward components: one CSV per component (raw + smooth)
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_f_dist.csv"), t, comp_f_dist, "f_dist")
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_f_dist_smooth_{SMOOTH_WINDOW}.csv"), t_s, comp_f_dist_s, "f_dist_smooth")

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_f_head_pos.csv"), t, comp_f_head_pos, "f_head_pos")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_comp_f_head_pos_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    comp_f_head_pos_s,
    "f_head_pos_smooth",
)

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_f_head_vel.csv"), t, comp_f_head_vel, "f_head_vel")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_comp_f_head_vel_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    comp_f_head_vel_s,
    "f_head_vel_smooth",
)

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_w_dist.csv"), t, comp_w_dist, "w_dist")
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_w_dist_smooth_{SMOOTH_WINDOW}.csv"), t_s, comp_w_dist_s, "w_dist_smooth")

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_w_head_pos.csv"), t, comp_w_head_pos, "w_head_pos")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_comp_w_head_pos_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    comp_w_head_pos_s,
    "w_head_pos_smooth",
)

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_w_head_vel.csv"), t, comp_w_head_vel, "w_head_vel")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_comp_w_head_vel_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    comp_w_head_vel_s,
    "w_head_vel_smooth",
)

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_bias.csv"), t, comp_bias, "bias")
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_bias_smooth_{SMOOTH_WINDOW}.csv"), t_s, comp_bias_s, "bias_smooth")

save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_comp_total_reward_mean.csv"), t, comp_total_reward, "total_reward_mean")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_comp_total_reward_mean_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    comp_total_reward_s,
    "total_reward_mean_smooth",
)

plot_raw_and_smooth(
    out_png=os.path.join(fig_dir, f"{ALGO}_training_reward.png"),
    out_pdf=os.path.join(fig_dir, f"{ALGO}_training_reward.pdf"),
    timesteps_raw=t,
    values_raw=train_reward,
    timesteps_smooth=t_s,
    values_smooth=reward_s,
    xlabel="Timesteps",
    ylabel="Reward",
    title="Training Reward (episode return)",
)

plot_raw_and_smooth(
    out_png=os.path.join(fig_dir, f"{ALGO}_final_distance.png"),
    out_pdf=os.path.join(fig_dir, f"{ALGO}_final_distance.pdf"),
    timesteps_raw=t,
    values_raw=final_distance,
    timesteps_smooth=t_s,
    values_smooth=dist_s,
    xlabel="Timesteps",
    ylabel="Final Distance",
    title="Final Distance (episode end)",
)

plot_raw_and_smooth(
    out_png=os.path.join(fig_dir, f"{ALGO}_totally_behind.png"),
    out_pdf=os.path.join(fig_dir, f"{ALGO}_totally_behind.pdf"),
    timesteps_raw=t,
    values_raw=totally_behind,
    timesteps_smooth=t_s,
    values_smooth=behind_s,
    xlabel="Timesteps",
    ylabel="Totally Behind (count)",
    title="Totally Behind (episode count)",
)

plot_components_smooth(
    out_png=os.path.join(fig_dir, f"{ALGO}_reward_components_smooth.png"),
    out_pdf=os.path.join(fig_dir, f"{ALGO}_reward_components_smooth.pdf"),
    timesteps_smooth=t_s,
    comps_smooth={
        "w_dist": comp_w_dist_s,
        "w_head_pos": comp_w_head_pos_s,
        "w_head_vel": comp_w_head_vel_s,
        "bias": comp_bias_s,
        "total_reward_mean": comp_total_reward_s,
    },
    xlabel="Timesteps",
    ylabel="Component value",
    title=f"Reward Components (smooth_{SMOOTH_WINDOW})",
)

manifest = {
    "algo": ALGO,
    "total_timesteps": TOTAL_TIMESTEPS,
    "smooth_window": SMOOTH_WINDOW,
    "k_history": K_HISTORY,
    "run_dir": run_dir,
    "model_path": model_path,
    "csv_dir": csv_dir,
    "fig_dir": fig_dir,
}
with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nTraining completed!")
print(f"Saved model to: {model_path}")
print(f"Saved CSVs to:   {csv_dir}")
print(f"Saved figures to:{fig_dir}")