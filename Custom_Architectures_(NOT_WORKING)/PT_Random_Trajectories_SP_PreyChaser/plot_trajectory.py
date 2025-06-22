import numpy as np
import matplotlib.pyplot as plt
import env_PT_rand_traj as env

# Trajectory parameters
bounds = np.array([[0, 10.0], [0, 10.0], [0, 10.0]], dtype=np.float32)
dt = 0.01
initial_speed = 1.0
max_step = initial_speed * dt
T = 500
K = 20 # Number of waypoints

# Generate and plot
traj = env.generate_bspline_trajectory(bounds, max_step, dt, T=T, K=K)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Traiettoria 3D generata con B-spline')
plt.show()
