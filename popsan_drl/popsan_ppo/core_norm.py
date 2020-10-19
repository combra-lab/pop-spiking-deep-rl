import torch
import torch.nn as nn
from torch import sqrt
from torch.distributions.normal import Normal
import sys
sys.path.append("../../")

from popsan_drl.popsan_ppo.popsan import PopSpikeActor


class CriticNet(nn.Module):
    """Critic network: can use for Q Net and V Net"""
    def __init__(self, network_shape, state_shape):
        """
        :param network_shape: list of hidden layer sizes
        :param state_shape: shape of state
        :param action_shape: shape of action
        """
        super(CriticNet, self).__init__()
        layer_num = len(network_shape)
        self.model = [nn.Linear(state_shape, network_shape[0]),
                      nn.ReLU()]
        if layer_num > 1:
            for layer in range(layer_num-1):
                self.model.extend(
                    [nn.Linear(network_shape[layer], network_shape[layer+1]),
                     nn.ReLU()])
        self.model.extend([nn.Linear(network_shape[-1], 1)])
        self.model = nn.Sequential(*self.model)

    def forward(self, state):
        out = self.model(state)
        return out


class SpikeActorDeepCritic(nn.Module):
    def __init__(self, state_shape, action_shape, encoder_pop_dim,
                 decoder_pop_dim, mean_range, std, spike_ts, device,
                 hidden_size):
        super(SpikeActorDeepCritic, self).__init__()
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1
        self.device = device

        self.critic = CriticNet(hidden_size, state_shape)
        self.popsan = PopSpikeActor(state_shape, action_shape, encoder_pop_dim, decoder_pop_dim,
                                    hidden_size, mean_range, std, spike_ts, device)

    def forward(self, x, batch_size):
        value = self.critic(x)
        mu, std = self.popsan(x, batch_size)
        dist = Normal(mu, std)
        return dist, value

    def normalize_state(self, state, update=True):
        """
        Use Welford's algorithm to normalize a state, and optionally update the statistics
        for normalizing states using the new state, online.
        """
        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1)).to(self.device)
            self.welford_state_mean_diff = torch.ones(state.size(-1)).to(self.device)

        if update:
            if len(state.size()) == 1:  # If we get a single state vector
                state_old = self.welford_state_mean
                self.welford_state_mean += (state - state_old) / self.welford_state_n
                self.welford_state_mean_diff += (state - state_old) * (state - state_old)
                self.welford_state_n += 1
            else:
                raise RuntimeError
        return (state - self.welford_state_mean) / sqrt(self.welford_state_mean_diff / self.welford_state_n)

    def copy_normalizer_stats(self, net):
        self.welford_state_mean = net.self_state_mean
        self.welford_state_mean_diff = net.welford_state_mean_diff
        self.welford_state_n = net.welford_state_n
