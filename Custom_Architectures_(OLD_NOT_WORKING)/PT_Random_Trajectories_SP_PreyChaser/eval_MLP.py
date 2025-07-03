import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import env_PT_rand_traj

class PredictiveModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.feature = nn.Sequential(*layers)
        self.head = nn.Linear(last, 3)

    def forward(self, x):
        return self.head(self.feature(x))

def evaluate(model_path, n_trajectories=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load model
    model = PredictiveModel(input_dim=5).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    dt = env_PT_rand_traj.PointMassEnv.dt
    T  = env_PT_rand_traj.PointMassEnv.max_steps
    initial_speed = 1.0
    max_step = initial_speed * dt
    bounds = np.array([
        [0, +10.0],
        [0, +10.0],
        [0, +10.0]
    ], dtype=np.float32)

    mse_list = []
    mae_list = []
    mape_list = []

    for _ in range(n_trajectories):
        traj = env_PT_rand_traj.generate_bspline_trajectory(bounds, max_step, dt, T=T, K=6)
        # build state (current) and next_pos (ground truth) arrays
        states = np.zeros((T-1, 5), dtype=np.float32)
        next_pos = np.zeros((T-1, 3), dtype=np.float32)
        for t in range(T-1):
            x, y, z = traj[t]
            dx, dy, dz = traj[t+1] - traj[t]
            theta = np.arctan2(dy, dx)
            v = np.linalg.norm([dx, dy, dz]) / dt
            states[t]   = [x, y, z, v, theta]
            next_pos[t] = traj[t+1]

        # prediction
        with torch.no_grad():
            inp = torch.from_numpy(states).to(device)
            preds = model(inp).cpu().numpy()

        err = preds - next_pos
        mse = np.mean(err**2)
        mae = np.mean(np.abs(err))
        mape = np.mean(np.abs(err / next_pos)) * 100
        mse_list.append(mse)
        mae_list.append(mae)
        mape_list.append(mape)

    print(f"Evaluated on {n_trajectories} trajectories:")
    print(f"Mean MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
    print(f"Mean MAE: {np.mean(mae_list):.6f} ± {np.std(mae_list):.6f}")
    print(f"Mean MAPE: {np.mean(mape_list):.6f}% ± {np.std(mape_list):.6f}%")


model_name = "predictive_20250510_161358.pth"
LOAD_PATH = os.path.join("models", model_name)
NUM = 10000

evaluate(model_path=LOAD_PATH, n_trajectories=NUM)