import torch
import numpy as np
import time
import gym
import pickle
import math
import sys

sys.path.append("../../")
from popsan_drl.popsan_td3.replay_buffer_norm import ReplayBuffer
from popsan_drl.popsan_td3.popsan import PopSpikeActor
from loihi_realization.popsan_loihi import SpikingActorNet
from loihi_realization.utility import read_pytorch_network_parameters_4_loihi, \
    combine_multiple_into_one_int, decoder_multiple_from_one_int


def test_mujoco_render(popsan_model_file, env_fn, encoder_pop_dim, decoder_pop_dim, mean_range, std, spike_ts,
                       buffer_mean, buffer_var, hidden_size=(256, 256), norm_clip_limit=3):
    """
    Test and render Mujoco tasks
    :param popsan_model_file: file dir for popsan model
    :param env_fn: function of create environment
    :param encoder_pop_dim: encoder population dimension
    :param decoder_pop_dim: decoder population dimension
    :param mean_range: mean range for encoder
    :param std: std for encoder
    :param spike_ts: spike timesteps
    :param buffer_mean: mean for Replay Buffer
    :param buffer_var: var for Replay Buffer
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
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=1,
                                 clip_limit=norm_clip_limit, norm_update_every=1)
    replay_buffer.mean = buffer_mean
    replay_buffer.var = buffer_var
    # PopSAN
    popsan = PopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_size,
                           mean_range, std, spike_ts, act_limit, device, False)
    popsan.load_state_dict(torch.load(popsan_model_file))
    en_mean, en_var, l_w, l_b, de_w, de_b = read_pytorch_network_parameters_4_loihi(popsan)
    # Set core list in a special format easier for reset layer by layer
    encoder_core_list = [0 for _ in range(obs_dim * encoder_pop_dim)]
    bias_core_list = [1 for _ in range(len(hidden_size) + 1)]
    decoder_core_list = [2 for _ in range(act_dim * decoder_pop_dim)]
    hidden_layer_core_list = [[num // 128 + 3 + layer*2 for num in range(hidden_size[layer])] for layer in range(len(hidden_size))]
    core_list = [encoder_core_list]
    core_list.extend(hidden_layer_core_list)
    core_list.extend([decoder_core_list, bias_core_list])
    # Setup PopSAN on Loihi
    popsan_loihi = SpikingActorNet(l_w, l_b, core_list)
    board, en_channel, de_channel = popsan_loihi.setup_snn(44, 6)
    # Start Loihi SNN Simulation
    board.startDriver()
    board.run(1000 * (spike_ts + 4), aSync=True)
    # Begin Test

    def gen_compact_pop_act(pop_act):
        """
        Generate compact population activity
        :param pop_act: population activity
        :return: compact_pop_act, compact_num
        """
        compact_pop_act = combine_multiple_into_one_int(pop_act[:168])
        compact_pop_act.extend(pop_act[168:])
        return compact_pop_act, 44

    def get_action(o):
        """
        Generate action
        :param o: observation
        :return: action
        """
        start_time = time.time()
        # Generate packed Input Neuron Activity
        o = o.reshape(obs_dim, 1)
        en_pop_act = np.exp(-(1. / 2.) * (o - en_mean)**2 / en_var).reshape(obs_dim * encoder_pop_dim)
        en_pop_act = np.int_(en_pop_act * 100)
        en_pop_act = en_pop_act.tolist()
        compact_en_pop_act, compact_num = gen_compact_pop_act(en_pop_act)
        # Write and read with Loihi
        en_channel.write(compact_num, compact_en_pop_act)
        compact_de_pop_act = de_channel.read(6)
        # Decoder packed Output Neuron Activity to Action
        de_pop_act = decoder_multiple_from_one_int(compact_de_pop_act, num_bits=3, overall_bits=30)
        de_pop_act = np.array(de_pop_act) / spike_ts
        de_pop_act = de_pop_act.reshape(act_dim, decoder_pop_dim)
        action = (de_pop_act * de_w.squeeze()).sum(axis=1).squeeze() + de_b.squeeze()
        action = np.tanh(action)
        end_time = time.time()
        return action, end_time - start_time

    # Test One Episode
    o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    overall_time = 0
    while not (d or (ep_len == 1000)):
        # test_env.render()
        act, time_use = get_action(replay_buffer.normalize_obs(o))
        o, r, d, _ = test_env.step(act)
        overall_time += time_use
        ep_ret += r
        ep_len += 1
    board.finishRun()
    board.disconnect()
    print("Inf/s: ", 1 / (overall_time / 1000))
    print("Reward: ", ep_ret)


def test_best_models(env_name, snip_name, model_num, result_dir, spike_ts=5, encoder_pop_dim=10, decoder_pop_dim=10,
                     std=math.sqrt(0.15)):
    """
    Test best models from all trained trails
    :param env_name: environment name
    :param snip_name: directory to snip
    :param model_num: number models in saved file
    :param result_dir: directory to result
    :param spike_ts: spike timesteps
    :param encoder_pop_dim: encoder population dimension
    :param decoder_pop_dim: decoder population dimension
    :param std: std for encoder
    :return:
    """
    test_env = gym.make(env_name)
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.shape[0]
    act_limit = test_env.action_space.high[0]
    neuron_input_memory = []
    input_channel_data_shape = (obs_dim * 10) // 4 + (obs_dim * 10) % 4
    # Set core list in a special format easier for reset layer by layer
    if env_name != "Ant-v3":
        encoder_core_list = [0 for _ in range(obs_dim * encoder_pop_dim)]
        bias_core_list = [1 for _ in range(3)]
        decoder_core_list = [2 for _ in range(act_dim * decoder_pop_dim)]
        hidden_layer_core_list = [[num // 128 + 3 + layer * 2 for num in range(256)] for layer in range(2)]
    else:
        encoder_core_list = [num // 222 for num in range(obs_dim * encoder_pop_dim)]
        bias_core_list = [5 for _ in range(3)]
        decoder_core_list = [6 for _ in range(act_dim * decoder_pop_dim)]
        hidden_layer_core_list = [[num // 32 + 7 + layer * 8 for num in range(256)] for layer in range(2)]
    core_list = [encoder_core_list]
    core_list.extend(hidden_layer_core_list)
    core_list.extend([decoder_core_list, bias_core_list])

    def gen_compact_pop_act(pop_act):
        """
        Generate compact population activity
        :param pop_act: population activity
        :return: compact_pop_act, compact_num
        """
        compact_idx = obs_dim * 10 - (obs_dim * 10) % 4
        compact_pop_act = combine_multiple_into_one_int(pop_act[:compact_idx])
        compact_pop_act.extend(pop_act[compact_idx:])
        return compact_pop_act

    def get_action(o):
        """
        Generate action
        :param o: observation
        :return: action
        """
        start_time = time.time()
        # Generate packed Input Neuron Activity
        o = o.reshape(obs_dim, 1)
        en_pop_act = np.exp(-(1. / 2.) * (o - en_mean)**2 / en_var).reshape(obs_dim * encoder_pop_dim)
        en_pop_act = np.int_(en_pop_act * 100)
        en_pop_act = en_pop_act.tolist()
        compact_en_pop_act = gen_compact_pop_act(en_pop_act)
        # Write and read with Loihi
        neuron_input_memory.append(compact_en_pop_act)
        en_channel.write(input_channel_data_shape, compact_en_pop_act)
        compact_de_pop_act = de_channel.read(act_dim)
        # Decoder packed Output Neuron Activity to Action
        de_pop_act = decoder_multiple_from_one_int(compact_de_pop_act, num_bits=3, overall_bits=30)
        de_pop_act = np.array(de_pop_act) / spike_ts
        de_pop_act = de_pop_act.reshape(act_dim, decoder_pop_dim)
        action = (de_pop_act * de_w.squeeze()).sum(axis=1).squeeze() + de_b.squeeze()
        action = np.tanh(action)
        end_time = time.time()
        return action, end_time - start_time

    reward_list = np.zeros(model_num)
    inference_list = np.zeros(model_num)
    for m in range(model_num):
        test_reward, _ = pickle.load(open(result_dir + '/model' + str(m) + '_test_rewards.p', 'rb'))
        best_epoch_idx = 0
        best_epoch_reward = 0
        for idx in range(20):
            if test_reward[(idx + 1) * 5 - 1] > best_epoch_reward:
                best_epoch_reward = test_reward[(idx + 1) * 5 - 1]
                best_epoch_idx = (idx + 1) * 5
        print("Train Model: ", m, " Best Epoch: ", best_epoch_idx, " Reward: ", best_epoch_reward)
        model_dir = result_dir + '/model' + str(m) + '_e' + str(best_epoch_idx) + '.pt'
        buffer_dir = result_dir + '/model' + str(m) + '_e' + str(best_epoch_idx) + '_mean_var.p'
        # Set device
        device = torch.device("cpu")
        # Replay buffer for running z-score norm
        b_mean_var = pickle.load(open(buffer_dir, "rb"))
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=1,
                                     clip_limit=3, norm_update_every=1)
        replay_buffer.mean = b_mean_var[0]
        replay_buffer.var = b_mean_var[1]
        # PopSAN
        popsan = PopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, (256, 256),
                               (-3, 3), std, spike_ts, act_limit, device, False)
        popsan.load_state_dict(torch.load(model_dir))
        en_mean, en_var, l_w, l_b, de_w, de_b = read_pytorch_network_parameters_4_loihi(popsan)
        # Setup PopSAN on Loihi
        popsan_loihi = SpikingActorNet(l_w, l_b, core_list)
        board, en_channel, de_channel = popsan_loihi.setup_snn(input_channel_data_shape, act_dim, snip_dir=snip_name)
        # Start Loihi SNN Simulation
        board.startDriver()
        board.run(10 * 1000 * (spike_ts + 4), aSync=True)
        # Begin Test
        single_model_reward_list = np.zeros(10)
        single_model_inf_list = np.zeros(10)
        for e in range(10):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            overall_time = 0
            while not (d or (ep_len == 1000)):
                act, time_use = get_action(replay_buffer.normalize_obs(o))
                o, r, d, _ = test_env.step(act)
                overall_time += time_use
                ep_ret += r
                ep_len += 1
            while not (ep_len == 1000):
                _, time_use = get_action(replay_buffer.normalize_obs(o))
                overall_time += time_use
                ep_len += 1
            single_model_reward_list[e] = ep_ret
            single_model_inf_list[e] = 1 / (overall_time / ep_len)
        board.finishRun()
        board.disconnect()
        reward_list[m] = np.mean(single_model_reward_list)
        inference_list[m] = np.mean(single_model_inf_list)
        print("Model: ", m, " Avg Reward: ", reward_list[m], " Avg Inf/s: ", inference_list[m])
    return reward_list, inference_list, neuron_input_memory


if __name__ == '__main__':
    data_dir = "<Dir to saved models>"
    env_name = "HalfCheetah-v3"
    snip_name = "./snip_halfcheetah"
    model_num = 10
    r_list, i_list, in_mem = test_best_models(env_name, snip_name, model_num, data_dir)
    print("All Model Reward Mean: ", np.mean(r_list), " STD: ", np.std(r_list))
    print("All Model Inf/s Mean: ", np.mean(i_list), " STD: ", np.std(i_list))
    print("Length of input memory: ", len(in_mem))
