import numpy as np
import copy


def pytorch_trained_snn_param_2_loihi_snn_param(weight, bias, vth):
    """
    Transform pytorch weights, bias, and vth based on the scale
    :param weight: pytorch weights
    :param bias: pytorch bias
    :param vth: pytorch vth
    :return: weights_dict, new_vth, scale_factor
    """
    max_w = np.amax(weight)
    min_w = np.amin(weight)
    max_b = np.amax(bias)
    min_b = np.amin(bias)
    '''
    First find the scale factor
    Method:
        1. Find max absolute value between [max_w, min_w, max_b, min_b]
        2. Then the scale factor is 255 divide the max absolute value
    '''
    max_abs_value = max(abs(max_w), abs(min_w), abs(max_b), abs(min_b))
    scale_factor = 255. / max_abs_value
    '''
    Second Compute New Loihi Voltage Threshold
    '''
    new_vth = int(vth * scale_factor)
    '''
    Third Compute Scaled Loihi Weight and Bias
    '''
    new_w = np.clip(weight * scale_factor, -255, 255)
    new_w = new_w.astype(int)
    new_b = np.clip(bias * scale_factor, -255, 255)
    new_b = new_b.astype(int)
    pos_w = copy.deepcopy(new_w)
    pos_w[new_w < 0] = 0
    pos_w_mask = np.int_(new_w > 0)
    neg_w = copy.deepcopy(new_w)
    neg_w[new_w > 0] = 0
    neg_w_mask = np.int_(new_w < 0)
    pos_b = copy.deepcopy(new_b)
    pos_b[new_b < 0] = 0
    pos_b_mask = np.int_(new_b > 0)
    pos_b = pos_b.reshape((-1, 1))
    pos_b_mask = pos_b_mask.reshape((-1, 1))
    neg_b = copy.deepcopy(new_b)
    neg_b[new_b > 0] = 0
    neg_b_mask = np.int_(new_b < 0)
    neg_b = neg_b.reshape((-1, 1))
    neg_b_mask = neg_b_mask.reshape((-1, 1))
    '''
    Generate weights_dict and return
    '''
    weights_dict = {'pos_w': pos_w,
                    'neg_w': neg_w,
                    'pos_w_mask': pos_w_mask,
                    'neg_w_mask': neg_w_mask,
                    'pos_b': pos_b,
                    'neg_b': neg_b,
                    'pos_b_mask': pos_b_mask,
                    'neg_b_mask': neg_b_mask}
    return weights_dict, new_vth, scale_factor


def read_pytorch_network_parameters_4_loihi(network, use_learn_encoder=True):
    """
    Read parameters from pytorch network
    :param network: pytorch popsan
    :param use_learn_encoder: if true read from network with learn encoder
    :return: encoder mean and var, hidden layer weights and biases, decoder weights and biases
    """
    if use_learn_encoder:
        encoder_mean = network.encoder.mean.data.numpy()
        encoder_var = network.encoder.std.data.numpy()
        encoder_var = encoder_var**2
    else:
        encoder_mean = network.encoder.mean.numpy()
        encoder_var = network.encoder.var
    layer_weights, layer_bias = [], []
    for i, fc in enumerate(network.snn.hidden_layers, 0):
        tmp_weights = fc.weight.data.numpy()
        tmp_bias = fc.bias.data.numpy()
        layer_weights.append(tmp_weights)
        layer_bias.append(tmp_bias)
    layer_weights.append(network.snn.out_pop_layer.weight.data.numpy())
    layer_bias.append(network.snn.out_pop_layer.bias.data.numpy())
    decoder_weights = network.decoder.decoder.weight.data.numpy()
    decoder_bias = network.decoder.decoder.bias.data.numpy()
    return encoder_mean, encoder_var, layer_weights, layer_bias, decoder_weights, decoder_bias


def combine_multiple_into_one_int(input_list, num_bits=7, overall_bits=28):
    """
    Combine multiple integers into one integer to save space
    :param input_list: list of input integers
    :param num_bits: max number of bits for item in input list
    :param overall_bits: overall bits of one integer
    :return: encode_list
    """
    int_per_int_num = overall_bits // num_bits
    assert (len(input_list) % int_per_int_num) == 0
    encode_list = []
    encode_list_num = len(input_list) // int_per_int_num
    for num in range(encode_list_num):
        start_num = num * int_per_int_num
        end_num = (num + 1) * int_per_int_num
        big_int = 0
        for i, small_int in enumerate(input_list[start_num:end_num], 0):
            big_int = big_int + (small_int << (i * num_bits))
        encode_list.append(big_int)
    return encode_list


def decoder_multiple_from_one_int(encode_list, num_bits=7, overall_bits=28):
    """
    Decode one integer into multiple integers
    :param encode_list: list of encoded integers
    :param num_bits: max number of bits for item in input list
    :param overall_bits: overall bits of one integer
    :return: input_list
    """
    input_list = []
    int_per_int_num = overall_bits // num_bits
    for big_int in encode_list:
        tmp_big_int = big_int
        for i in range(int_per_int_num):
            small_int = tmp_big_int - (tmp_big_int >> num_bits << num_bits)
            tmp_big_int = tmp_big_int >> num_bits
            input_list.append(small_int)
    return input_list

