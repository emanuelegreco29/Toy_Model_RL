import copy
import numpy as np
import torch
from .models import EnhancedActorCritic

def other(agent, agent_names):
    return agent_names[1] if agent == agent_names[0] else agent_names[0]

class OpponentPool:
    """
    Mantiene snapshot e statistiche (wins/games). Il campionamento è 
    'prioritized fictitious self-play': preferisce avversari con winrate
    contro di noi vicino al 60-70% (quindi leggermente più forti).
    """
    def __init__(self, agent_name, obs_dim, act_dim, device):
        self.agent_name = agent_name
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.snapshots = []   # list of state_dict
        self.stats = []       # list of dict: {'wins': int, 'games': int}

    def add(self, state_dict):
        self.snapshots.append(copy.deepcopy(state_dict))
        self.stats.append({'wins': 0, 'games': 0})

    def _score(self, wr):
        # priorizza WR ~0.65 (più forte ma battibile). 0..1
        return float(np.exp(-((wr - 0.65) / 0.20) ** 2))

    def sample_actor(self, mix_self_prob: float = 0.1):
        """
        Ritorna (idx, actor). Con prob mix_self_prob usa None per indicare
        'gioca contro la policy corrente' (self-play puro).
        """
        if len(self.snapshots) == 0 or np.random.rand() < mix_self_prob:
            return None, None

        wrs = []
        for st in self.stats:
            g = max(1, st['games'])
            wrs.append(st['wins'] / g)

        scores = np.array([self._score(wr) for wr in wrs], dtype=np.float32)
        # un pizzico di esplorazione
        probs = scores + 0.05
        probs = probs / probs.sum()

        idx = int(np.random.choice(len(self.snapshots), p=probs))
        actor = EnhancedActorCritic(self.obs_dim, self.act_dim).to(self.device)
        actor.load_state_dict(self.snapshots[idx])
        actor.eval()
        return idx, actor

    def record_result(self, idx, did_we_win: bool):
        if idx is None or idx >= len(self.stats):
            return
        self.stats[idx]['games'] += 1
        self.stats[idx]['wins']  += 1 if did_we_win else 0