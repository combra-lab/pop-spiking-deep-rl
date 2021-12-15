import numpy as np
import torch
import torch.nn as nn
import gym
import pickle
import math
import sys

sys.path.append("../../")
from popsan_drl.popsan_td3.replay_buffer_norm import ReplayBuffer
from popsan_drl.popsan_td3.popsan import PopSpikeActor


def test_mujoco_render(popsan_model_file, mean_var_file, env_fn,
                       encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts,
                       hidden_size=(256, 256), norm_clip_limit=3):
    """
    Test and render Mojuco Tasks

    :param popsan_model_file: file dir for popsan model
    :param mean_var_file: file dir for mean and var of replay buffer
    :param env_fn: function of create environment
    :param encoder_pop_dim: encoder population dimension
    :param decoder_pop_dim: decoder population dimension
    :param mean_range: mean range for encoder
    :param std: std for encoder
    :param spike_ts: spike timesteps
    :param hidden_size: list of hidden layer sizes
    :param norm_clip_limit: clip limit
    """
    # Set device
    device = torch.device("cpu")
    # Set environment
    test_env = env_fn()
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.shape[0]
    act_limit = test_env.action_space.high[0]

    # Replay buffer for running z-score norm
    b_mean_var = pickle.load(open(mean_var_file, "rb"))
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=1,
                                 clip_limit=norm_clip_limit, norm_update_every=1)
    replay_buffer.mean = b_mean_var[0]
    replay_buffer.var = b_mean_var[1]

    # PopSAN
    popsan = PopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_size,
                           mean_range, std, spike_ts, act_limit, device, False)
    popsan.load_state_dict(torch.load(popsan_model_file))

    def get_action(o):
        a = popsan(torch.as_tensor(o, dtype=torch.float32, device=device), 1).numpy()
        return np.clip(a, -act_limit, act_limit)

    # Start testing
    o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    while not (d or (ep_len == 1000)):
        test_env.render()
        with torch.no_grad():
            o, r, d, _ = test_env.step(get_action(replay_buffer.normalize_obs(o)))
        ep_ret += r
        ep_len += 1
    print("Reward: ", ep_ret)


if __name__ == '__main__':
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file_dir', type=str, default='./params/spike-td3_td3-popsan-HalfCheetah-v3-encoder-dim-10-decoder-dim-10/model0_e100.pt')
    parser.add_argument('--buffer_file_dir', type=str, default='./params/spike-td3_td3-popsan-HalfCheetah-v3-encoder-dim-10-decoder-dim-10/model0_e100_mean_var.p')
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    args = parser.parse_args()
    mean_range = (-3, 3)
    std = math.sqrt(args.encoder_var)
    spike_ts = 5

    test_mujoco_render(args.model_file_dir, 
                       args.buffer_file_dir, 
                       lambda : gym.make(args.env), 
                       args.encoder_pop_dim, 
                       args.decoder_pop_dim, 
                       mean_range, std, spike_ts)

