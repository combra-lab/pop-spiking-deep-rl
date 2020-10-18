#include <stdlib.h>
#include "nxsdk.h"
#include "decoder.h"
#define out_num 30 // Different for different Mujoco Task

extern int epoch;

int de_out_spikes[out_num] = {0};

// Compact Output: Different for different spiking timesteps
// (T=5, bits=3), (T=15, bits=4), (T=25, bits=5)
int de_decode_num_bits = 3;
int de_decode_overall_bits = 30;
int de_decode_per_int_num = 10;

// Reset Cores determined by core list setup
int de_reset_core_start = 2;
int de_reset_core_end = 6;

static int numNeuronsPerCore = 1024;
static int NUM_Y_TILES = 5;

int do_decoder(runState *s)
{
    return 1;
}

void run_decoder(runState *s)
{
    int time = s->time_step;
    /*
    Counting output spikes
    */
    if(time % epoch > 4 || time % epoch == 0)
    {
        for(int i=0; i<out_num; i++)
        {
            if(SPIKE_COUNT[(time)&3][i+0x20] > 0)
            {
                de_out_spikes[i] += 1;
            }
            SPIKE_COUNT[(time)&3][i+0x20] = 0;
        }
    }
    /*
    Generate compact output list and send to host
    */
    if(time % epoch == 0)
    {
        int OutputChannelId = getChannelID("decodeoutput");
        int encode_list[3] = {0};
        for(int i=0; i<3; i++)
        {
            int start_num = i * de_decode_per_int_num;
            int end_num = (i + 1) * de_decode_per_int_num;
            int big_int = 0;
            for(int j=start_num; j<end_num; j++)
            {
                int shift_bits = (j - start_num) * de_decode_num_bits;
                big_int = big_int + (de_out_spikes[j] << shift_bits);
            }
            encode_list[i] = big_int;
        }
        writeChannel(OutputChannelId, encode_list, 3);
        for(int i=0; i<out_num; i++)
        {
            de_out_spikes[i] = 0;
        }
    }
    /*
    Reset each layer separately at the end of operation
    */
    if(time % epoch == 7 || time % epoch == 8)
    {
        core_hard_reset(3, 4);
    }
    if(time % epoch == 8 || time % epoch == 0)
    {
        core_hard_reset(5, 6);
    }
    if(time % epoch == 0 || time % epoch == 1)
    {
        core_hard_reset(2, 2);
    }
}

void core_hard_reset(int start, int end)
{
    NeuronCore *nc;
    CoreId coreId;

    CxState cxs = (CxState) {.U=0, .V=0};
    for(int i=start; i<end+1; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init64(nc->cx_state, numNeuronsPerCore, *(uint64_t*)&cxs);
    }

    for(int i=start; i<end+1; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init32(nc->dendrite_accum, 8192/2, 0);
    }

    MetaState ms = (MetaState) {.Phase0=2, .SomaOp0=3,
                                .Phase1=2, .SomaOp1=3,
                                .Phase2=2, .SomaOp2=3,
                                .Phase3=2, .SomaOp3=3};
    for(int i=start; i<end+1; i++) {
        logicalToPhysicalCoreId(i, &coreId);
        nc = NEURON_PTR(coreId);
        nx_fast_init32(nc->cx_meta_state, numNeuronsPerCore/4, *(uint32_t*)&ms);
    }
}

void logicalToPhysicalCoreId(int logicalId, CoreId *physicalId)
{
    physicalId->p = logicalId % 4;
    physicalId->x = logicalId/(4*(NUM_Y_TILES-1));
    physicalId->y = (logicalId - physicalId->x*4*(NUM_Y_TILES-1))/4 + 1;
}