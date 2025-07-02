import numpy as np

def compute_reward(self):
        agent_pos = self.state[:3]
        yaw, pitch = self.state[4:6]
        target_pos = self.target_state
        target_dir = self._get_target_direction()
        
        vec = agent_pos - target_pos
        d3 = np.linalg.norm(vec) + 1e-8
        f_dist = np.exp(-0.5 * (d3/10) ** 1)

        ux, uy, uz = vec / d3
        position_vec = np.array([ux, uy, uz])
        
        dx = np.cos(pitch) * np.cos(yaw)
        dy = np.cos(pitch) * np.sin(yaw)
        dz = np.sin(pitch)
        direction_vec = np.array([dx, dy, dz])
        
        cos_vel = np.clip(np.dot(direction_vec, target_dir), -1.0, 1.0)
        cos_pos = np.clip(np.dot(position_vec, target_dir), -1.0, 1.0)
        
        f_head_pos = (1 - cos_pos) / 2
        f_head_vel = (1 + cos_vel) / 2

        return (f_dist * 0.5 + f_head_pos * 0.3 + f_head_vel * 0.2) - 1.0