#ifndef CAR_LIB_CARFIELD_H
#define CAR_LIB_CARFIELD_H

#include <stdint.h>

#include "match/ctx.h"

#include "carfield_lib/dma.h"

#define L1_SCRATCHPAD_SIZE 32768


// General host functions

void carfield_init();

void carfield_init_uart();

void carfield_shutdown();

void handle_host_dma_transfer(
    void* src,
    void* dst,
    size_t size
);

void carfield_timer_start();
uint64_t carfield_timer_stop();

// Host functions specific for pulp_cluster exec module

void pulp_cluster_reset();

void pulp_cluster_offload_async(void* boot_addr);
void pulp_cluster_offload_blk(void* boot_addr);

extern const uint8_t __l2_common_start[];
extern const uint8_t __l2_common_end[];

#define offload_args ((volatile uint32_t*)__l2_common_start)


#endif // CAR_LIB_CARFIELD_H