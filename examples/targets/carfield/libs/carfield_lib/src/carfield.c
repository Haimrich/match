#include "carfield_lib/carfield.h"

#include <stdint.h>
#include <string.h>

// Carfield
#include "car_util.h"
#include "car_memory_map.h"
#include "regs/system_timer.h"
// Cheshire
#include "dif/clint.h"
#include "dif/dma.h"
#include "params.h"
#include "regs/cheshire.h"
#include "util.h"
// Carfield Lib
#include "carfield_lib/uart.h"


#define VERIFY_DMA 0
#define CARFIELD_DEBUG 0


void carfield_init() {
    // Initialize the Carfield SoC
    car_enable_domain(CAR_PULP_RST);
    // Initialize the UART
    carfield_init_uart();

    mini_printf("Hi, there. I'm Carfield 🐱\r\n\n");
}

void carfield_shutdown() {
    car_disable_domain(CAR_PULP_RST);
    mini_printf("\r\nBye.\r\n");
}


void pulp_cluster_reset() {
    volatile uint32_t *booten_addr = (uint32_t*)(CAR_INT_CLUSTER_BOOTEN_ADDR(car_soc_ctrl));
    writew(0, booten_addr);
    volatile uint32_t *fetchen_addr = (uint32_t*)(CAR_INT_CLUSTER_FETCHEN_ADDR(car_soc_ctrl));
    writew(0, fetchen_addr);
    volatile uint32_t *pulp_eoc_addr = (uint32_t*)(CAR_INT_CLUSTER_EOC_ADDR(car_soc_ctrl));
    writew(0, pulp_eoc_addr);
    pulp_cluster_set_bootaddress(0);
    car_reset_domain(CAR_PULP_RST);
}


void pulp_cluster_offload_async(void* boot_addr)
{
    #if CARFIELD_DEBUG
    mini_printf("Starting PULP cluster...\r\n");
    #endif
    pulp_cluster_reset();
    pulp_cluster_set_bootaddress(boot_addr);
    pulp_cluster_start();
    //mini_printf("> Started PULP cluster. Waiting...\r\n");
}


void pulp_cluster_offload_blk(void* boot_addr)
{
    pulp_cluster_offload_async(boot_addr);
    pulp_cluster_wait_eoc();
    #if CARFIELD_DEBUG
    mini_printf("> Cluster finished.\r\n");
    #endif
}


// Other things

void handle_host_dma_transfer(void* src, void* dst, size_t size) 
{
    #if CARFIELD_DEBUG
    mini_printf("Starting DMA transfer...\r\n");
    #endif
    sys_dma_2d_blk_memcpy(dst, src, size, 0, 0, 1);
    #if CARFIELD_DEBUG
    mini_printf("Transfer complete.\r\n");
    #endif
    #if VERIFY_DMA
    // Verify
    volatile uint8_t* src_ptr = (uint8_t*)src;
    volatile uint8_t* dst_ptr = (uint8_t*)dst;
    bool transfer_success = true;
    for (int i = 0; i < size; i++) {
        volatile uint8_t sval = src_ptr[i];
        volatile uint8_t dval = dst_ptr[i];
        if (sval != dval) {
            mini_printf("DMA transfer failed at byte %d\r\n", i);
            mini_printf("src_ptr %p -> %d\r\n", src_ptr + i, sval);
            mini_printf("dst_ptr %p -> %d\r\n", dst_ptr + i, dval);
            transfer_success = false;
            break;
        }
    }
    if (transfer_success)  {
        mini_printf("Transfer Verified Successfully.\r\n");
    }
    #endif
}


void carfield_init_uart() {
    uint32_t rtc_freq = *reg32(&__base_regs, CHESHIRE_RTC_FREQ_REG_OFFSET);
    uint64_t reset_freq = clint_get_core_freq(rtc_freq, 2500);
    car_uart_init(&__base_uart, reset_freq, 115200);
}


void carfield_timer_start() {
    writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_RESET_LO_OFFSET);
    writed(1, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_START_LO_OFFSET);
}
    
uint64_t carfield_timer_stop() {
    writed(0, CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CFG_LO_OFFSET);
    volatile uint64_t time = readd(CAR_SYSTEM_TIMER_BASE_ADDR + TIMER_CNT_LO_OFFSET);
    return time;
}