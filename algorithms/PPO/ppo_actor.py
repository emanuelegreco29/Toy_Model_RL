import torch
import torch.nn as nn

from ..utilities.mlp import MLPBase
from ..utilities.gru import GRULayer
from ..utilities.action_layer import ACTLayer
from ..utilities.utilities import convert


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        # (1) feature extraction module
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        input_size = self.base.output_size
        
        # (2) rnn module
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id)

        self.apply(self._init_weights)

        self.to(device)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = convert(obs).to(**self.tpdv)
        rnn_states = convert(rnn_states).to(**self.tpdv)
        masks = convert(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = convert(obs).to(**self.tpdv)
        rnn_states = convert(rnn_states).to(**self.tpdv)
        action = convert(action).to(**self.tpdv)
        masks = convert(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = convert(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy