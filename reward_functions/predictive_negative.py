import numpy as np

def compute_reward(self, prev_agent: np.ndarray, cur_agent: np.ndarray) -> float:
        # Distance improvement towards current target
        cur_target = self.target_history[-1]
        prev_dist = np.linalg.norm(prev_agent[:3] - cur_target[:3])
        cur_dist  = np.linalg.norm(cur_agent[:3]  - cur_target[:3])
        imp = prev_dist - cur_dist
        if imp > 0:
            reward = imp
        else:
            reward = -0.5 * abs(imp)

        # Tracking bonus
        if(self.tracking_counter > 0):
            reward += 1.0 * self.tracking_counter

        history = np.array(self.target_history)
        pred_next = self._predict_target(history, horizon=5)
        prev_pd = np.linalg.norm(prev_agent[:3] - pred_next)
        cur_pd  = np.linalg.norm(cur_agent[:3]  - pred_next)
        pred_imp = prev_pd - cur_pd
        reward += 0.1 * pred_imp

        # agent_v and agent_yaw_rate
        agent_v = cur_agent[3]
        # Delta yaw cur and prev
        delta_yaw_agent = (cur_agent[4] - prev_agent[4]) / self.dt

        # target_speed, target_acc and target_yaw_rate
        hist = np.array(self.target_history)
        deltas = np.diff(hist, axis=0)             # (K,3)
        speeds = np.linalg.norm(deltas, axis=1)    # (K,)
        vels   = deltas / self.dt                  # (K,3)
        accs   = np.diff(speeds) / self.dt         # (K-1,)
        headings = np.arctan2(deltas[:,1], deltas[:,0])
        yaw_rates = np.diff(headings) / self.dt

        # Use latest values
        tgt_speed    = speeds[-1]    if len(speeds)>0    else 0.0
        tgt_acc      = accs[-1]      if len(accs)>0      else 0.0
        tgt_yaw_rate = yaw_rates[-1] if len(yaw_rates)>0 else 0.0

        # Slow down if target slows down
        dv_target = tgt_acc * self.dt
        dv_agent  = agent_v - prev_agent[3]

        # if dv_target>0) no reward
        if dv_target < 0:
            # error is normalized
            err_v = abs(dv_agent - dv_target) / 0.5
            vel_bonus = max(0.0, 1.0 - err_v)  # 1 if perfect, 0 if error>=0.5
        else:
            vel_bonus = 0.0

        # Reward if predict change of direction
        err_yaw = abs(delta_yaw_agent - tgt_yaw_rate)
        # normalization
        yaw_bonus = max(0.0, 1.0 - err_yaw / np.pi)

        # only positive rewards (or 0)
        reward += 0.2   * vel_bonus
        reward += 0.2 * yaw_bonus

        # vettore agente→target proiettato in XY
        vec = self.target_state[:2] - cur_agent[:2]
        dist_xy = np.linalg.norm(vec) + 1e-6
        dir_xy = vec / dist_xy

        # agent XY heading
        yaw = cur_agent[4]
        head_xy = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

        # coseno dell’angolo tra dove guardo e dove devo andare
        alignment = np.dot(head_xy, dir_xy)

        # premiamo fino a +0.2 se alignment==1, 0 se ortogonale
        reward += 0.2 * max(0.0, alignment)
        #reward = (1.0 / (1.0 + np.exp(-reward))) - 1.0 # Sigmoid per normalizzazione

        return float(reward)