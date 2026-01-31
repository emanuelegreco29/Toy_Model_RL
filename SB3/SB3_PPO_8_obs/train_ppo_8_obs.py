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

from env_sb3_ppo_8_obs import PointMassEnv

ALGO = "SAC"  # "SAC"  # "TD3"

TOTAL_TIMESTEPS = 500 * 500
SMOOTH_WINDOW = 25


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
    total_improvement: float
    comp_improvement_term: float
    comp_step_penalty: float
    comp_success_bonus: float
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
            total_improvement = float(info.get("total_improvement", np.nan))

            rc = info.get("reward_components", {}) or {}
            comp_improvement_term = float(rc.get("improvement_term", np.nan))
            comp_step_penalty = float(rc.get("step_penalty", np.nan))
            comp_success_bonus = float(rc.get("success_bonus", np.nan))
            comp_total_reward = float(rc.get("total_reward", np.nan))

            self.records.append(
                EpisodeRecord(
                    timestep=num_ts,
                    ep_reward=ep_reward,
                    final_distance=final_distance,
                    total_improvement=total_improvement,
                    comp_improvement_term=comp_improvement_term,
                    comp_step_penalty=comp_step_penalty,
                    comp_success_bonus=comp_success_bonus,
                    comp_total_reward=comp_total_reward,
                )
            )

            n = len(self.records)
            print(
                f"Episode {n} | Timestep {num_ts} | Reward: {ep_reward:.2f} | "
                f"FinalDist: {final_distance:.2f} | TotalImprovement: {total_improvement:.2f}"
            )

        return True


# 1) env SB3
env = DummyVecEnv([lambda: Monitor(PointMassEnv())])

# 2) modello
if ALGO == "PPO":
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        ent_coef=0.02,
        clip_range=0.2,
        n_epochs=10,
        batch_size=64,
        verbose=0,
    )
elif ALGO == "SAC":
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        batch_size=256,
        tau=0.005,
        verbose=0,
    )
elif ALGO == "TD3":
    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        batch_size=256,
        tau=0.005,
        verbose=0,
    )
else:
    raise ValueError(f"Unknown ALGO={ALGO}")

# 3) callback logger
callback = EpisodeLogger()

# 4) training
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# 5) crea run_dir e salva modello lì dentro
ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("models", f"{ALGO.lower()}_sb3_8_obs_{ts}")
ensure_dir(run_dir)
ensure_dir(os.path.join(run_dir, "figures"))
ensure_dir(os.path.join(run_dir, "csv"))

model_path = os.path.join(run_dir, f"{ALGO.lower()}_model.zip")
model.save(model_path)

# 6) prepara arrays
t = np.array([r.timestep for r in callback.records], dtype=np.int64)

train_reward = np.array([r.ep_reward for r in callback.records], dtype=float)
final_distance = np.array([r.final_distance for r in callback.records], dtype=float)
total_improvement = np.array([r.total_improvement for r in callback.records], dtype=float)

comp_improvement_term = np.array([r.comp_improvement_term for r in callback.records], dtype=float)
comp_step_penalty = np.array([r.comp_step_penalty for r in callback.records], dtype=float)
comp_success_bonus = np.array([r.comp_success_bonus for r in callback.records], dtype=float)
comp_total_reward = np.array([r.comp_total_reward for r in callback.records], dtype=float)

# smoothing (valid -> timesteps accorciati)
reward_s = moving_average(train_reward, SMOOTH_WINDOW)
dist_s = moving_average(final_distance, SMOOTH_WINDOW)
impr_s = moving_average(total_improvement, SMOOTH_WINDOW)

comp_improvement_term_s = moving_average(comp_improvement_term, SMOOTH_WINDOW)
comp_step_penalty_s = moving_average(comp_step_penalty, SMOOTH_WINDOW)
comp_success_bonus_s = moving_average(comp_success_bonus, SMOOTH_WINDOW)
comp_total_reward_s = moving_average(comp_total_reward, SMOOTH_WINDOW)

t_s = t[SMOOTH_WINDOW - 1 :] if t.size >= SMOOTH_WINDOW else np.array([], dtype=np.int64)

# 7) salva CSV (raw e smooth) stile LaTeX
csv_dir = os.path.join(run_dir, "csv")

# training reward
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_reward.csv"), t, train_reward, "reward")
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_reward_smooth_{SMOOTH_WINDOW}.csv"), t_s, reward_s, "reward_smooth")

# final distance
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_final_distance.csv"), t, final_distance, "final_distance")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_final_distance_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    dist_s,
    "final_distance_smooth",
)

# total improvement
save_csv_two_cols(os.path.join(csv_dir, f"{ALGO}_PointMass_total_improvement.csv"), t, total_improvement, "total_improvement")
save_csv_two_cols(
    os.path.join(csv_dir, f"{ALGO}_PointMass_total_improvement_smooth_{SMOOTH_WINDOW}.csv"),
    t_s,
    impr_s,
    "total_improvement_smooth",
)

# reward components (raw multi-col)
components_raw_path = os.path.join(csv_dir, f"{ALGO}_PointMass_reward_components.csv")
with open(components_raw_path, "w", encoding="utf-8") as f:
    f.write("timestep,improvement_term,step_penalty,success_bonus,total_reward\n")
    for i in range(t.size):
        f.write(
            f"{int(t[i])},"
            f"{float(comp_improvement_term[i])},"
            f"{float(comp_step_penalty[i])},"
            f"{float(comp_success_bonus[i])},"
            f"{float(comp_total_reward[i])}\n"
        )

# reward components (smooth multi-col)
components_s_path = os.path.join(csv_dir, f"{ALGO}_PointMass_reward_components_smooth_{SMOOTH_WINDOW}.csv")
with open(components_s_path, "w", encoding="utf-8") as f:
    f.write("timestep,improvement_term_smooth,step_penalty_smooth,success_bonus_smooth,total_reward_smooth\n")
    for i in range(t_s.size):
        f.write(
            f"{int(t_s[i])},"
            f"{float(comp_improvement_term_s[i])},"
            f"{float(comp_step_penalty_s[i])},"
            f"{float(comp_success_bonus_s[i])},"
            f"{float(comp_total_reward_s[i])}\n"
        )

# 8) salva figure (PNG+PDF) nella directory del modello
fig_dir = os.path.join(run_dir, "figures")

plot_raw_and_smooth(
    out_png=os.path.join(fig_dir, f"{ALGO}_training_reward.png"),
    out_pdf=os.path.join(fig_dir, f"{ALGO}_training_reward.pdf"),
    timesteps_raw=t,
    values_raw=train_reward,
    timesteps_smooth=t_s,
    values_smooth=reward_s,
    xlabel="Timesteps",
    ylabel="Reward",
    title="Training Reward",
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
    out_png=os.path.join(fig_dir, f"{ALGO}_total_improvement.png"),
    out_pdf=os.path.join(fig_dir, f"{ALGO}_total_improvement.pdf"),
    timesteps_raw=t,
    values_raw=total_improvement,
    timesteps_smooth=t_s,
    values_smooth=impr_s,
    xlabel="Timesteps",
    ylabel="Total Improvement",
    title="Total Improvement (start_dist - final_dist)",
)

plot_components_smooth(
    out_png=os.path.join(fig_dir, f"{ALGO}_reward_components_smooth.png"),
    out_pdf=os.path.join(fig_dir, f"{ALGO}_reward_components_smooth.pdf"),
    timesteps_smooth=t_s,
    comps_smooth={
        "improvement_term": comp_improvement_term_s,
        "step_penalty": comp_step_penalty_s,
        "success_bonus": comp_success_bonus_s,
        "total_reward": comp_total_reward_s,
    },
    xlabel="Timesteps",
    ylabel="Component value",
    title=f"Reward Components (smooth_{SMOOTH_WINDOW})",
)

# 9) salva un piccolo manifest per ricordarti dove stanno le cose
manifest = {
    "algo": ALGO,
    "total_timesteps": TOTAL_TIMESTEPS,
    "smooth_window": SMOOTH_WINDOW,
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
