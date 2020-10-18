#include <stdlib.h>
#include "nxsdk.h"
#include "encoder.h"
#define input_num 1110 // Different for different Mujoco Task
#define bias_num 3

int epoch = 5 + 4; // Base epoch time 5 can be changed to other spiking timesteps (15, 25)

int en_act[input_num] = {0};

int en_spike_v[input_num] = {0};
int en_spike_vth = 99;

int en_input_core_start = 0;
int en_input_neuron_per_core = 222;
int en_input_chip = 0;
int en_input_start = 0;

int en_bias_core = 5;
int en_bias_chip = 0;
int en_bias_start = 0;

int en_decode_num_bits = 7;
int en_decode_overall_bits = 28;
int en_decode_per_int_num = 4;

int do_encoder(runState *s)
{
    return 1;
}

void run_encoder(runState *s)
{
    int time = s->time_step;
    /*
    Read and unpack neuron activity from host
    (The length of data and the format is determined by the observation space of Mujoco task)
    */
    if(time % epoch == 1)
    {
        int InputChannelId = getChannelID("encodeinput");
        int tmp_input[279];
        readChannel(InputChannelId, &tmp_input, 279);
        int decode_output_idx = 0;
        for(int i=0; i<277; i++)
        {

            int big_int = tmp_input[i];
            for(int j=0; j<en_decode_per_int_num; j++)
            {
                en_act[decode_output_idx] = big_int - ((big_int >> en_decode_num_bits) << en_decode_num_bits);
                en_spike_v[decode_output_idx] = en_act[decode_output_idx];
                big_int = big_int >> en_decode_num_bits;
                decode_output_idx++;
            }
        }
        en_act[1108] = tmp_input[277];
        en_spike_v[1108] = en_act[1108];
        en_act[1109] = tmp_input[278];
        en_spike_v[1109] = en_act[1109];
    }
    /*
    Generate spikes base on the neuron activity
    */
    if(time % epoch > 0 && time % epoch <= 5)
    {
        // Generate Regular Spikes for Encoder Population
        for(int i=0; i<input_num; i++)
        {
            if(en_spike_v[i] > en_spike_vth)
            {
                en_spike_v[i] -= en_spike_vth;
                int input_core_id = en_input_core_start + i / en_input_neuron_per_core;
                int input_axon_id = en_input_start + i % en_input_neuron_per_core;
                uint16_t axonId = 1<<14 | ((input_axon_id) & 0x3FFF);
                ChipId chipId = nx_nth_chipid(en_input_chip);
                nx_send_remote_event(time, chipId, (CoreId){.id=4+input_core_id}, axonId);
            }
            en_spike_v[i] += en_act[i];
        }
    }
    // Inject Spikes to Bias Neurons
    for(int i=0; i<bias_num; i++)
    {
        if(time % epoch > i && time % epoch <= i + 5)
        {
            int bias_axon_id = en_bias_start + i;
            uint16_t axonId = 1<<14 | ((bias_axon_id) & 0x3FFF);
            ChipId chipId = nx_nth_chipid(en_bias_chip);
            nx_send_remote_event(time, chipId, (CoreId){.id=4+en_bias_core}, axonId);
        }
    }
}
