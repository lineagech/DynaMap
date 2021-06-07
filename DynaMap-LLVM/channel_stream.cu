#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <atomic>
#include <assert.h>
#include <chrono>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "inc.def"

#define ULL unsigned long long int

#define CUDA_SAFECALL(call)                                                 \
{                                                                       \
    call;                                                               \
    cudaError err = cudaGetLastError();                                 \
    if (cudaSuccess != err) {                                           \
        fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
        fflush(stderr);                                                 \
        _exit(EXIT_FAILURE);                                            \
    }                                                                   \
}

#if 1
#define cudaPinned(data,bytes)\
do {\
	cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);\
    cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, 0);\
} while(0)
#define cudaPinToGpu(data,bytes)\
do {\
	cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, 0);\
    cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, 0);\
    cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);\
} while(0)
#else
#define cudaPinned(data,bytes)
#endif

#define cudaPinnedGpu(data,bytes)\
do {\
	cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, 0);\
    cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);\
    \
} while(0)

#define StatInGPU 1
#define RAW_TRANSFER 0
#define INSTRUMENTED 1
#define PINNED_BUFF_SIZE ((ULL)1024*1024*256)
#define MEM_FENCE 1

#define MULTI_BUFFERS 0 
#define NUM_BUFFERS 1

#define PROFILE_RUN 0

#define DEBUG_PRT(fmt, ...) printf(fmt, ##__VA_ARGS__)
//#define DEBUG_PRT(fmt, ...)

#define PAGE_SIZE_BITS (12)
#define BLOCK_SIZE_BITS (21)
#define PAGE_SIZE (0x1<<PAGE_SIZE_BITS)

#define PAGE_ALIGN(x) (x & ~((0x1<<PAGE_SIZE_BITS)-1))
#define BLOCK_ALIGN(x) (x & ~((0x1<<BLOCK_SIZE_BITS)-1))

#define UPDT_WARP_NUM 4
#define UPDT_BLOCK_NUM 4
#define THREADS_IN_WARP 32
#define ENT_PER_PAGE (PAGE_SIZE/QUANT_LINE_SIZE)

#define DIV_ROUND_UP(n,d) (((n) + (d) - 1) / (d))
#define DECLARE_BITMAP(name, bits) uint64_t name[DIV_ROUND_UP(bits, sizeof(uint64_t))]

#define GET_PAGE_OFFSET(_addr) (_addr & (PAGE_SIZE-1))
#define GET_SUB_PAGE_OFFSET(_addr) (_addr & (QUANT_LINE_SIZE*8-1))

#define ASSIGN_BUFF_START_END(buff, _num)\
{\
    uint8_t* tmp_ptr = buff;\
    for (int buff_idx = 0; buff_idx < _num; buff_idx++) {\
        buff_start[buff_idx] = tmp_ptr;\
        buff_end[buff_idx] = tmp_ptr + buff_size - 1;\
        tmp_ptr += buff_size;\
        printf("%d: [%lx,%lx]\n", buff_idx, (uint64_t)buff_start[buff_idx], (uint64_t)buff_end[buff_idx]);\
    }\
}

#define NOT_NEED_UPDATE (0xFFFFFFFFFFFFFFFF)

#define TEMPORAL_REUSE_MEASURE 0
//#define INCREMENT_PROFLE 1

#if TEMPORAL_REUSE_MEASURE
#define NO_NEED_UPDATE 0xFFFFFFFF
#else
#define NO_NEED_UPDATE 0xFF
#endif

struct MemAccInfo {
    void* _addr;
    //uint8_t _size;
};

typedef struct {
    uint64_t page_aligned_start_addr;
    uint64_t page_aligned_end_addr;
    uint64_t region_size;
    uint64_t inst_en;
    #if TEMPORAL_REUSE_MEASURE 
    int*     stat_table;
    #else
    uint8_t* stat_table;
    #endif
} MemAccStatTable_t;

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(ret));
    return ret;
}

/* Lock Implementation */
#if TEMPORAL_REUSE_MEASURE 
__device__ volatile int sem = 0;
__device__ void acquire_sem(volatile int* lock) 
{
    while (atomicCAS((int*)lock, 0, 1) != 0);
}
__device__ void release_sem(volatile int* lock) 
{
    *lock = 0;
    __threadfence();
}

#define ENTER_CRITICAL()
#define EXIT_CRITICAL()

/*
#define ENTER_CRITICAL() \
do {\
    __syncthreads();\
    if (threadIdx.x == 0) acquire_sem(&sem);\
    __syncthreads();\
} while(0);

#define EXIT_CRITICAL() \
do {\
    __threadfence();\
    __syncthreads();\
    if (threadIdx.x == 0) release_sem(&sem);\
    __syncthreads();\
} while(0);
*/
#endif

class ChannelDev {
private:
    int id;
    int doorbell_idx;
    volatile int* doorbell;
    
    uint8_t *buff;
    uint8_t *buff_start[NUM_BUFFERS];
    uint8_t *buff_end[NUM_BUFFERS];

    uint8_t *volatile buff_write_head_ptr;
    uint8_t *volatile buff_write_tail_ptr;

    uint64_t buff_size;
    uint32_t num_buff;
    
public:
    /* Stats Table */
    uint32_t page_stat_table_size;
    MemAccStatTable_t* page_stat_table;
    
    ChannelDev() {}
    
    __device__ __forceinline__ int get_PST_index_given_addr (uint64_t accessed_addr)
    {
        int table_index = -1;
        for (int i = 0; i < page_stat_table_size; i++) {
            if (accessed_addr >= page_stat_table[i].page_aligned_start_addr && 
                    accessed_addr < page_stat_table[i].page_aligned_end_addr) {
                return i;
            }
        } 
        return table_index;
    }
    
    __device__ __forceinline__ void push_pst_2d_thread_y (uint64_t address, int typeSize, int scale_factor, int y_dist)  
    {
        int table_index;
        int page_index;
        int line_index;

        /* based on scale factor, calculate the exact address where next warp or block accesses */
        for (int i = 0; i < (UPDT_BLOCK_NUM/y_dist); i++) { 
            table_index = get_PST_index_given_addr(address); 
            if (table_index == -1) return;
            page_index = (address - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
            line_index = GET_PAGE_OFFSET(address) >> QUANT_LINE_SIZE_BITS;
            /* for now we just gather spatial locality to get spatial utilization,
               so just setting to 1 rather than accumulating */ 
            if (page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] != NO_NEED_UPDATE) {
                page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] = 1;
            }    
            address += (scale_factor * typeSize) * y_dist;
        }
    }
    
    
    __device__ __forceinline__ void push_pst_2d_y (uint64_t address, int typeSize, int scale_factor, uint64_t row_size)  
    {
        int table_index;
        int page_index;
        int line_index;
        int block_page_index;
        int block_idx = blockIdx.y;
        int block_dim = gridDim.y;

        /* based on scale factor, calculate the exact address where next warp or block accesses */
        for (int i = 0; i < 32; i++) { 
            table_index = get_PST_index_given_addr(address); 
            if (table_index == -1) return;
            page_index = (address - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
            
            if (BLOCK_ALIGN(address) >= page_stat_table[table_index].page_aligned_start_addr) 
                block_page_index = (BLOCK_ALIGN(address) - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
            else 
                block_page_index = 0;

            if (page_stat_table[table_index].stat_table[block_page_index*ENT_PER_PAGE] == NO_NEED_UPDATE) {
                //DEBUG_PRT("%s: table[%d] block[%d] no need to update\n", __func__, table_index, block_page_index>>(BLOCK_SIZE_BITS-PAGE_SIZE_BITS));
                return;
            }

            line_index = GET_PAGE_OFFSET(address) >> QUANT_LINE_SIZE_BITS;
            
            #if TEMPORAL_REUSE_MEASURE
            ENTER_CRITICAL();
            //atomicAdd(&(page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index]), 1);
            page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] += 1;
            EXIT_CRITICAL();
            #else
            /* for now we just gather spatial locality to get spatial utilization,
               so just setting to 1 rather than accumulating */ 
            if (page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] == 0) {
                page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] = 1;
                //DEBUG_PRT("%s: wrten addr %lx === > address %lx ===> table index %u, page_index %u, line index %u\n",
                //    __func__, &(page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index]), address, table_index, page_index, line_index); 
            }
            #endif
            if (++block_idx >= block_dim) 
                break;

            address += (scale_factor * typeSize); 
        }

    }
    
    __device__ __forceinline__ void push_pst_2d_x (uint64_t address, int typeSize, int scale_factor)  
    {
        int table_index;
        int page_index;
        int line_index;
        int block_page_index;
        int block_idx = blockIdx.x;
        int block_dim = gridDim.x;

        /* based on scale factor, calculate the exact address where next warp or block accesses */
        for (int i = 0; i < 32; i++) { 
            table_index = get_PST_index_given_addr(address); 
            if (table_index == -1) return;
            page_index = (address - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
            
            if (BLOCK_ALIGN(address) >= page_stat_table[table_index].page_aligned_start_addr) 
                block_page_index = (BLOCK_ALIGN(address) - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
            else 
                block_page_index = 0;

            if (page_stat_table[table_index].stat_table[block_page_index*ENT_PER_PAGE] == NO_NEED_UPDATE) {
                //DEBUG_PRT("%s: table[%d] block[%d] no need to update\n", __func__, table_index, block_page_index>>(BLOCK_SIZE_BITS-PAGE_SIZE_BITS));
                return;
            }

            line_index = GET_PAGE_OFFSET(address) >> QUANT_LINE_SIZE_BITS;
            
            #if TEMPORAL_REUSE_MEASURE 
            ENTER_CRITICAL();
            //atomicAdd(&(page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index]), 1);
            page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] += 1;
            EXIT_CRITICAL();
            #else
            /* for now we just gather spatial locality to get spatial utilization,
               so just setting to 1 rather than accumulating */ 
            if (page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] == 0) {
                page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] = 1;
                //DEBUG_PRT("%s: wrten addr %lx === > address %lx ===> table index %u, page_index %u, line index %u\n",
                //    __func__, &(page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index]), address, table_index, page_index, line_index); 
            } 
            #endif
            if (++block_idx >= block_dim) 
                break;

            address += (scale_factor * typeSize); 
        }
    }

    __device__ __forceinline__ void push_pst_1d (uint64_t address, int typeSize, int scale_factor)  
    {
        int table_index;
        int page_index;
        int block_page_index;
        int line_index;
        
        /* based on scale factor, calculate the exact address where next warp or block accesses */
        for (int i = 0; i < UPDT_WARP_NUM; i++) { 
            table_index = get_PST_index_given_addr(address); 
            if (table_index == -1) return;

            page_index = (address - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
            
            if (BLOCK_ALIGN(address) >= page_stat_table[table_index].page_aligned_start_addr) 
                block_page_index = (BLOCK_ALIGN(address) - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
            else
                block_page_index = 0;

            //DEBUG_PRT("Check block aligned : %llx\n", &(page_stat_table[table_index].stat_table[block_page_index*ENT_PER_PAGE]));
            if (page_stat_table[table_index].stat_table[block_page_index*ENT_PER_PAGE] == NO_NEED_UPDATE) {
                //DEBUG_PRT("%s: table[%d] block[%d] no need to update\n", __func__, table_index, block_page_index>>(BLOCK_SIZE_BITS-PAGE_SIZE_BITS));
                return;
            }

            line_index = GET_PAGE_OFFSET(address) >> QUANT_LINE_SIZE_BITS;
           
            //DEBUG_PRT("stat table : %llx\n", &(page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index]));
            
            #if TEMPORAL_REUSE_MEASURE 
            ENTER_CRITICAL();
            //atomicAdd(&(page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index]), 1);
            page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] += 1;
            EXIT_CRITICAL();
            #else
            /* for now we just gather spatial locality to get spatial utilization,
               so just setting to 1 rather than accumulating */ 
            if (page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] == 0) {
                page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] = 1;
                //DEBUG_PRT("%s: wrten addr %lx === > address %lx ===> table index %u, page_index %u, line index %u\n",
                //    __func__, &(page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index]), address, table_index, page_index, line_index); 
            } 
            #endif
            //DEBUG_PRT("addr: %llx\n", address);
            address += (scale_factor * typeSize * THREADS_IN_WARP);
        }
    }

    __device__ __forceinline__ void push_page_stat(uint64_t address) 
    {
        int table_index;
        int page_index;
        int block_page_index;
        int line_index;

        /* based on scale factor, calculate the exact address where next warp or block accesses */
        table_index = get_PST_index_given_addr(address); 
        if (table_index == -1) return;
        page_index = (address - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
        
        if (BLOCK_ALIGN(address) >= page_stat_table[table_index].page_aligned_start_addr) 
            block_page_index = (BLOCK_ALIGN(address) - page_stat_table[table_index].page_aligned_start_addr) >> PAGE_SIZE_BITS;
        else
            block_page_index = 0;

        if (page_stat_table[table_index].stat_table[block_page_index*ENT_PER_PAGE] == NO_NEED_UPDATE) {
            return;
        }

        line_index = GET_PAGE_OFFSET(address) >> QUANT_LINE_SIZE_BITS;

        #if TEMPORAL_REUSE_MEASURE 
        ENTER_CRITICAL();
        //acquire_sem(&sem);
        //atomicAdd((int*)&(page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index]), 1);
        page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] += 1;
        __threadfence();
        //release_sem(&sem);
        EXIT_CRITICAL();
        #else
        /* for now we just gather spatial locality to get spatial utilization,
           so just setting to 1 rather than accumulating */ 
        if (page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] == 0) {
            page_stat_table[table_index].stat_table[page_index*ENT_PER_PAGE+line_index] = 1;
        }
        #endif
    }

    __device__ __forceinline__ void push(void* packet, uint32_t nbytes) 
    {
        uint8_t *curr_ptr = NULL;
        
        while (curr_ptr == NULL) {
            curr_ptr = (uint8_t*) atomicAdd((ULL*)&buff_write_head_ptr, (ULL)nbytes);

            /* handle buffer overflows */
            if (curr_ptr + nbytes > buff_end[0]) {
                if (curr_ptr <= buff_end[0]) {
                    //while (buff_write_tail_ptr != curr_ptr);
                    /* flush the buffer */
                    memcpy(curr_ptr, packet, nbytes);
                    flush();
                    return;
                }
                else {
                    /* waiting all write done */
                    while (buff_write_head_ptr > buff_end[0]);
                }
                curr_ptr = NULL;
            }
        }
        
        memcpy(curr_ptr, packet, nbytes);
        //DEBUG_PRT("%s: addr %lx, size %u\n", __func__, (uint64_t)((MemAccInfo*)packet)->_addr, ((MemAccInfo*)packet)->_size);
    }
    
    #if NUM_BUFFERS > 1
    __device__ __forceinline__ void push_multi_buffers(void* packet, uint32_t nbytes, int acc_num)
    {
        uint8_t *curr_ptr = NULL;
        size_t t_buf_idx = 0;
        uint8_t* t_buf_bound = 0;
        if ((threadIdx.x & 31) % acc_num != 0) {
            return;
        }
        while (curr_ptr == NULL) {
            curr_ptr = (uint8_t*) atomicAdd((ULL*)&buff_write_head_ptr, (ULL)nbytes);
            /* get index of buffers */
            t_buf_idx = (curr_ptr-buff) / buff_size + 1;
            t_buf_bound = ((uint64_t)t_buf_idx*(uint64_t)buff_size) + buff - 1;
            
            /* totally overflow, just wait 
               otherwise, directly write to the buffer */
            if (curr_ptr > buff_end[num_buff-1]) {
                while (buff_write_head_ptr > buff_end[num_buff-1]);
                curr_ptr = NULL;
                continue;
            }

            /* handle buffer overflows */
            if (curr_ptr + nbytes > t_buf_bound) {
                /* if the thread is the first one fulling the buffer, responsible for flushing */
                if (curr_ptr <= t_buf_bound) {
                    /* flush the buffer */
                    memcpy(curr_ptr, packet, nbytes);
                    flush_multi(t_buf_idx-1);
                    //DEBUG_PRT("Flush %d\n", t_buf_idx-1);
                    if (t_buf_idx == num_buff) {
                        buff_write_head_ptr = buff;
                        //DEBUG_PRT("Reset %lx\n", (uint64_t)buff_write_head_ptr);
                        //__threadfence();
                    }
                    break;
                }
            }
            else {
                //DEBUG_PRT("%lx %lx\n", (uint64_t)curr_ptr, (uint64_t)t_buf_bound);
                memcpy(curr_ptr, packet, nbytes);
            }
        }
    }
    
    __device__ __forceinline__ void flush_multi(uint32_t _idx)
    {
        /* ensure everything is visible */
        //__threadfence_system();
        
        doorbell[_idx] = (uint32_t)buff_size;;
        __threadfence_system();
    }

    #endif
    
    __device__ __forceinline__ void flush()
    {
        //uint32_t nbytes = (uint32_t)(buff_write_tail_ptr - buff);
        uint32_t nbytes = (uint32_t)buff_size;
        if (nbytes == 0) {
            return;
        }
        /* ensure everything is visible */
        __threadfence_system();
        //assert(*doorbell == 0);

        *doorbell = nbytes;
        //DEBUG_PRT("%s: total %u bytes\n", __func__, nbytes);
        __threadfence_system();

        while(*doorbell != 0);

        /* reset head/tail */
        buff_write_tail_ptr = buff;
        __threadfence();
        buff_write_head_ptr = buff;
    }
    
    int debug_var;
    __device__ void debug() {
        DEBUG_PRT("Var %d\n", debug_var);
    }

    __device__ void init_channel(void* _doorbell, 
                                 void* _buff_addr, 
                                 uint64_t _buff_size
                                 #if (NUM_BUFFERS > 0)
                                 , uint32_t _num_buff
                                 #endif
                                 ) {
        
        buff = (uint8_t*)_buff_addr;
        doorbell = (int*)_doorbell;
        buff_write_head_ptr = (uint8_t*)buff;
        buff_write_tail_ptr = (uint8_t*)buff;
        buff_size = _buff_size;
        
        #if (NUM_BUFFERS > 1)
        num_buff = _num_buff;
        ASSIGN_BUFF_START_END(buff, NUM_BUFFERS);
        doorbell_idx = 0;
        #else
        buff_end[0] = buff + buff_size - 1;
        num_buff = 1;
        #endif
        
        DEBUG_PRT("init channel: buff addr %lx, buff num %u\n", (uint64_t)buff, num_buff);
    }
    
    #if StatInGPU
    __device__ void init_stat_table(uint32_t _size, void* _stat_table) {
        page_stat_table_size = _size;
        page_stat_table = (MemAccStatTable_t*)_stat_table;
    }
    #endif

    friend class ChannelHost;
};

/* Declare receiving thread */
pthread_t collector_thread;
typedef void* (*thread_func)(void*); 

/* Buffer start and end pointer */
static uint8_t* buff = NULL;
#if NUM_BUFFERS > 1
static uint8_t* buff_start[NUM_BUFFERS] = {NULL};
static uint8_t* buff_end[NUM_BUFFERS] = {NULL};
static uint8_t* dev_buff_start[NUM_BUFFERS] = {NULL};
static uint8_t* dev_buff_end[NUM_BUFFERS] = {NULL};
#else
static uint8_t* buff_end = NULL;
#endif
static uint64_t buff_size = PINNED_BUFF_SIZE;

uint64_t *buff_write_head = NULL;
uint64_t *buff_write_tail = NULL;
int64_t *buff_curr_size = NULL;

/* Doorbell, notifying receiving thread */
#if NUM_BUFFERS > 1
static volatile int *doorbell = NULL;
#else
static volatile int *doorbell = NULL;
#endif

/* device related */
uint8_t* dev_buff = NULL;
uint8_t* dev_buff_read_head = NULL;

/* statistics */
std::chrono::time_point<std::chrono::high_resolution_clock> begin;
std::chrono::time_point<std::chrono::high_resolution_clock> end;

/* communication with uvm driver */
int uvm_fd = -1;

/* malloc managed arrays */
static MemAccStatTable_t* mallocManagedArr = NULL;
static MemAccStatTable_t* hostPageStatTable = NULL;
static int stat_table_size = 0;
static int stat_table_index = 0;

class ChannelHost {
private:
    cudaStream_t stream;
    int doorbell_idx = 0;

public:
    ChannelHost() {}
   
    void init(thread_func _recv_func) 
    {
        cudaDeviceProp prop;
        int device = 0;
        cudaGetDeviceProperties(&prop, device);
        if (prop.canMapHostMemory == 0) {
            CUDA_SAFECALL(cudaSetDeviceFlags(cudaDeviceMapHost));
        }

        DEBUG_PRT("GPU asyncEngineCount %d\n", prop.asyncEngineCount);
        
        int priority_high, priority_low;
        CUDA_SAFECALL(
            cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
        CUDA_SAFECALL(
            cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority_high));
         
    #if RAW_TRANSFER 
        
        /* create doorbell */
        CUDA_SAFECALL(cudaHostAlloc((void**)&doorbell, sizeof(int)*NUM_BUFFERS, cudaHostAllocMapped));
        #if NUM_BUFFERS > 1
        memset((void*)doorbell, 0, (size_t)sizeof(int)*NUM_BUFFERS);
        #else
        *doorbell = 0;
        #endif

        CUDA_SAFECALL(cudaMalloc((void**)&dev_buff, buff_size*NUM_BUFFERS))
        cudaPinnedGpu(dev_buff, buff_size*NUM_BUFFERS);
        dev_buff_read_head = dev_buff;
       
        assert(doorbell);
        assert(dev_buff);
    #endif
    }
    
    inline void get_page_stat()
    {   
        uint64_t _page_num;
        begin = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < stat_table_size; i++) {
            _page_num = (hostPageStatTable[i].region_size + (PAGE_SIZE-1))/PAGE_SIZE;
            cudaMemcpyAsync(hostPageStatTable[i].stat_table, dev_buff_read_head/*mallocManagedArr[i].stat_table*/, _page_num*(PAGE_SIZE/QUANT_LINE_SIZE),
                cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream); 
        
        end = std::chrono::high_resolution_clock::now();
        auto consumed_time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin);
        printf("Copy PST consumes %lu us with %lu bytes\n", consumed_time.count(), _page_num*(PAGE_SIZE/QUANT_LINE_SIZE));

        /*
        for (int i = 0; i < stat_table_size; i++) {
            int _page_num = (hostPageStatTable[i].region_size + (PAGE_SIZE-1))/PAGE_SIZE;
            for (int line_idx = 0; line_idx < _page_num*(PAGE_SIZE/QUANT_LINE_SIZE); line_idx++) {
                printf("Region %d - Page %d - Line %d : %d\n", i, 
                    line_idx/(PAGE_SIZE/QUANT_LINE_SIZE),
                    line_idx%(PAGE_SIZE/QUANT_LINE_SIZE),
                    hostPageStatTable[i].stat_table[line_idx]);
            }
        }
        */
    }

    uint32_t recv()
    {
        #if NUM_BUFFERS > 1
        uint32_t buff_nbytes = doorbell[doorbell_idx];
        #else
        uint32_t buff_nbytes = *doorbell;
        #endif
        if (buff_nbytes == 0) {
            return 0;
        }
        int nbytes = buff_nbytes;
        
        if (buff_nbytes > buff_size) {
            nbytes = buff_size;
        }
    
        //DEBUG_PRT("%s: copy the acc info back (%u bytes)\n", __func__, buff_nbytes);
        begin = std::chrono::high_resolution_clock::now();
        
        //cudaStreamSynchronize(stream);
        cudaMemcpyAsync(buff, dev_buff_read_head, nbytes, 
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        end = std::chrono::high_resolution_clock::now();
        auto int_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
        //DEBUG_PRT("Transfer time %lu usec with %d bytes\n", int_us.count(), nbytes);

        print(buff, nbytes);
        
        int bytes_left = buff_nbytes - nbytes;

        #if NUM_BUFFERS > 1
        dev_buff_read_head += nbytes;
        if (dev_buff_read_head > dev_buff_end[NUM_BUFFERS-1])
            dev_buff_read_head = dev_buff;
        doorbell[doorbell_idx] = 0;
        doorbell_idx = (doorbell_idx == NUM_BUFFERS-1) ? 0 : doorbell_idx+1;
        //DEBUG_PRT("Doorbell Index now is %u\n", doorbell_idx);
        #else
        dev_buff_read_head = dev_buff;
        *doorbell = 0;
        #endif

        //DEBUG_PRT("bytes_left %u\n", bytes_left);
        return nbytes;
    }
    
    void print(uint8_t* _info, int32_t _size)
    {
        uint8_t* info_ptr = _info;
        MemAccInfo* mem_acc_ptr = NULL;
        while (_size > 0){
            mem_acc_ptr = (MemAccInfo*)(info_ptr);
            //DEBUG_PRT("addr %lx, size %u\n", (uint64_t)mem_acc_ptr->_addr, mem_acc_ptr->_size);
            DEBUG_PRT("%lx\n", (uint64_t)mem_acc_ptr->_addr);
            _size -= sizeof(MemAccInfo);
            info_ptr += sizeof(MemAccInfo);
        }
    }

};

static ChannelDev channel_dev;
static ChannelHost channel_host;
static bool recv_thread_started = false;

void* channel_host_handle(void* arg)
{       
    DEBUG_PRT("%s\n", __func__);
    while (recv_thread_started) {
    #if StatInGPU
        //channel_host.get_page_stat(); 
    #endif
    #if RAW_TRANSFER        
        channel_host.recv(); 
    #endif
    }
    return NULL;
}

__device__ __noinline__ void deviceRecvWithLineAccNum(void* address, int typeSize, int line_access_num)
{
#if INSTRUMENTED 
    /* line access ratio means how many threads access the same line in the same warp */
    /* e.g. if 32 threads access the same line, the access_num would be 32 */
#if RAW_TRANSFER 
    MemAccInfo info = {
        ._addr = address, 
        //._size = (uint8_t)typeSize
    };
    #if NUM_BUFFERS > 1
    channel_dev.push_multi_buffers(&info, sizeof(MemAccInfo), line_access_num);
    #else
    channel_dev.push(&info, sizeof(MemAccInfo));
    #endif
#endif

#if StatInGPU
    if (!channel_dev.page_stat_table[0].inst_en) return;

    //int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    //if (((global_index >> 5) & 0x1) == 0) {
        channel_dev.push_page_stat((uint64_t)address);
    //}
    #if MEM_FENCE
    __threadfence_system();
    #endif

#endif

#endif 
    //__threadfence_system();
}


__device__ __noinline__ void deviceRecv(void* address, int typeSize, int acc_num)
{
#if INSTRUMENTED 
    /* line access ratio means how many threads access the same line in the same warp */
    /* e.g. if 32 threads access the same line, the access_num would be 32 */
#if RAW_TRANSFER 
    MemAccInfo info = {
        ._addr = address, 
        //._size = (uint8_t)typeSize
    };
    #if NUM_BUFFERS > 1
    channel_dev.push_multi_buffers(&info, sizeof(MemAccInfo), 1);
    #else
    channel_dev.push(&info, sizeof(MemAccInfo));
    #endif
#endif

#if StatInGPU
    if (!channel_dev.page_stat_table[0].inst_en) return;

    channel_dev.push_page_stat((uint64_t)address);
    #if MEM_FENCE
    __threadfence_system();
    #endif

#endif

#endif 
    //__threadfence_system();
}

__device__ void update_PST_2d_block_y (void* address, int typeSize, int scale_factor) 
{
#if INSTRUMENTED 
#if StatInGPU
    if (!channel_dev.page_stat_table[0].inst_en) return;

    //printf("addr %lx, type size %d, scale factor is %d\n", (uint64_t)address, typeSize, scale_factor);
    uint64_t row_size = blockDim.x*blockDim.y*gridDim.x;
    //uint64_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (((blockIdx.y >> 5) & (UPDT_WARP_NUM-1)) == 0) {
        //printf("update tid %lu\n", tid);
        channel_dev.push_pst_2d_y((uint64_t)address, typeSize, scale_factor, row_size);    
    }
#endif
#endif
}

__device__ void update_PST_2d_thread_y (void* address, int typeSize, int scale_factor) 
{
#if INSTRUMENTED 
#if StatInGPU
    if (!channel_dev.page_stat_table[0].inst_en) return;

    //printf("addr %lx, type size %d, scale factor is %d\n", (uint64_t)address, typeSize, scale_factor);
    int y_dist = (32 / blockDim.x);
    uint64_t row_size = blockDim.x*blockDim.y*gridDim.x;
    //uint64_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    
    if (y_dist == 0) y_dist = 1;

    if ((threadIdx.y & (UPDT_WARP_NUM-1)) == 0) {
        //printf("update tid %lu\n", tid);
       channel_dev.push_pst_2d_thread_y((uint64_t)address, typeSize, scale_factor, row_size);    
    }
#endif
#endif
}

__device__ void update_PST_2d_block_x(void* address, int typeSize, int scale_factor) 
{
#if INSTRUMENTED 
#if StatInGPU
    if (!channel_dev.page_stat_table[0].inst_en) return;

    //printf("addr %lx, type size %d, scale factor is %d\n", (uint64_t)address, typeSize, scale_factor);
    uint64_t block_id = blockIdx.x;
    if (((block_id >> 5) & (UPDT_WARP_NUM-1)) == 0) {
        channel_dev.push_pst_2d_x((uint64_t)address, typeSize, scale_factor);    
        #if MEM_FENCE
        __threadfence_system();
        #endif
    }
#endif
#endif
}

__device__ void update_PST_1d (void* address, int typeSize, int scale_factor) 
{
#if INSTRUMENTED 
#if StatInGPU
    //printf("addr %lx, type size %d, scale factor is %d\n", (uint64_t)address, typeSize, scale_factor);
    //return;
    if (!channel_dev.page_stat_table[0].inst_en) return;

    uint64_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (((tid >> 5) & (UPDT_WARP_NUM-1)) == 0) {
        //printf("update tid %lu\n", tid);
        channel_dev.push_pst_1d((uint64_t)address, typeSize, scale_factor);    
        //channel_dev.push_page_stat((uint64_t)address);
        #if MEM_FENCE
        __threadfence_system();
        #endif
    }
#endif

#if RAW_TRANSFER 
    MemAccInfo info = {
        ._addr = address, 
        //._size = (uint8_t)typeSize
    };
    #if NUM_BUFFERS > 1
    channel_dev.push_multi_buffers(&info, sizeof(MemAccInfo), 1);
    #else
    channel_dev.push(&info, sizeof(MemAccInfo));
    #endif
#endif

#endif
}

__global__ void psu_cuc1057_channel_device_init(uint64_t doorbell, 
                                                uint64_t buff_addr, 
                                                uint64_t buff_size
                                                #if (NUM_BUFFERS > 0)
                                                , uint32_t num_buffers
                                                #endif
                                                #if StatInGPU
                                                , uint64_t stat_table,
                                                uint32_t stat_table_size
                                                #endif
                                                )
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if (tid == 0) {
        #if RAW_TRANSFER 
        channel_dev.init_channel((void*)doorbell, 
                                 (void*)buff_addr, 
                                 buff_size
                                #if NUM_BUFFERS > 0
                                , num_buffers
                                #endif
                                );
        #endif
        #if StatInGPU
        DEBUG_PRT("%s\n", __func__);
        channel_dev.init_stat_table(stat_table_size, (void*)stat_table);
        #endif
    }
}

/* instrumenting function for stat table */
__host__ void mem_acc_stat_table_alloc(uint64_t* address, uint64_t alloc_size)
{
#if INSTRUMENTED && StatInGPU
    int _idx = stat_table_index++;
    int _page_num = (alloc_size + (PAGE_SIZE-1))/PAGE_SIZE;

    //DEBUG_PRT("%s: address %lx, size %lu, table size %lu\n", __func__, *address, alloc_size
    //    , _page_num*(PAGE_SIZE/QUANT_LINE_SIZE/sizeof(uint8_t))*sizeof(uint8_t));

    /* get page-aligned address of the managed area */
    //mallocManagedArr[_idx].page_aligned_start_addr = (uint64_t)(*address) /*& (~(PAGE_SIZE-1))*/;
    //mallocManagedArr[_idx].page_aligned_end_addr = ((uint64_t)(*address) + alloc_size) /*& (~(PAGE_SIZE-1))*/;
    //mallocManagedArr[_idx].region_size = alloc_size;
    /* allocate the space for page statistics table for the specific managed area */
    //CUDA_SAFECALL(cudaMallocManaged((void**)&(mallocManagedArr[_idx].stat_table), 
    //                _page_num*(PAGE_SIZE/QUANT_LINE_SIZE/sizeof(uint8_t))*sizeof(uint8_t)));
    /* pin the page statistics table on the device to avoid too many transactions between device and host */
    //cudaPinnedGpu(mallocManagedArr[_idx].stat_table, _page_num*(PAGE_SIZE/QUANT_LINE_SIZE/sizeof(uint8_t))*sizeof(uint8_t));
       
    //dev_buff_read_head = mallocManagedArr[_idx].stat_table;

    /* get page-aligned address of the managed area */
    hostPageStatTable[_idx].page_aligned_start_addr = (uint64_t)(*address) /*& (~(PAGE_SIZE-1))*/;
    hostPageStatTable[_idx].page_aligned_end_addr = ((uint64_t)(*address) + alloc_size) /*& (~(PAGE_SIZE-1))*/;
    hostPageStatTable[_idx].inst_en = 1;
    hostPageStatTable[_idx].region_size = alloc_size;
    /* allocate the space for page statistics table for the specific managed area */
    uint64_t quant_table_page_aligned_size = (((uint64_t)_page_num*(PAGE_SIZE/QUANT_LINE_SIZE) + (PAGE_SIZE-1)) >> PAGE_SIZE_BITS) << PAGE_SIZE_BITS;
    #if TEMPORAL_REUSE_MEASURE   
    quant_table_page_aligned_size *= 4;
    #endif
    DEBUG_PRT("stat table %d alloc size %lu\n", _idx, _page_num*(PAGE_SIZE/QUANT_LINE_SIZE));
    cudaMallocHost((void**)&(hostPageStatTable[_idx].stat_table), 
    //cudaMallocManaged((void**)&(hostPageStatTable[_idx].stat_table), 
                    quant_table_page_aligned_size/*_page_num*(PAGE_SIZE/QUANT_LINE_SIZE)*/);
    /* pin the page statistics table on the device to avoid too many transactions between device and host */
    //cudaPinned(hostPageStatTable[_idx].stat_table, _page_num*(PAGE_SIZE/QUANT_LINE_SIZE));
    DEBUG_PRT("host page stat table[%d] address %lx, size %lu, quant_stat_table %lx\n", 
              _idx, 
              (uint64_t)&(hostPageStatTable[_idx]), 
              (uint64_t)_page_num*(PAGE_SIZE/QUANT_LINE_SIZE),
              (uint64_t)hostPageStatTable[_idx].stat_table);
    
    stat_table_size++;

    // tmp
    //memset(hostPageStatTable[_idx].stat_table, 9, _page_num*(PAGE_SIZE/QUANT_LINE_SIZE));

    uvm_params _params = { (uint64_t)_idx, quant_table_page_aligned_size };
    ioctl(uvm_fd, UVM_MAP_QUANT_TABLE, &_params);
    
    
    #if PROFILE_RUN
    uvm_params _params_profile = { (uint64_t)(*address), ((uint64_t)(*address) + alloc_size)};
    ioctl(uvm_fd, UVM_PROFILE_ADDR, &_params_profile);
    #endif

#endif
}

__host__ void channel_host_init(int numMallocManaged)
{
#if INSTRUMENTED 

#if RAW_TRANSFER
    //CUDA_SAFECALL(cudaMallocManaged((void**)&buff, buff_size*NUM_BUFFERS));
    CUDA_SAFECALL(cudaHostAlloc((void**)&buff, buff_size*NUM_BUFFERS, cudaHostAllocMapped));
    assert(buff);
    //cudaPinned(buff, buff_size*NUM_BUFFERS);

    #if NUM_BUFFERS > 1
    ASSIGN_BUFF_START_END(buff, NUM_BUFFERS);
    #else
    buff_end = buff + buff_size - 1;
    #endif    
    
    channel_host.init(&channel_host_handle);
#endif
    
    DEBUG_PRT("num malloc managed %d, &hostPageStatTable %lx\n", numMallocManaged, &(hostPageStatTable));

#if NUM_BUFFERS > 0    
    //ASSIGN_BUFF_START_END(dev_buff, NUM_BUFFERS);
#endif

#if StatInGPU
    /* for malloc managed memory */

    //CUDA_SAFECALL(cudaMallocManaged((void**)&mallocManagedArr, numMallocManaged*sizeof(MemAccStatTable_t)));
    //cudaPinnedGpu(mallocManagedArr, numMallocManaged*sizeof(MemAccStatTable_t));
    CUDA_SAFECALL(cudaMallocManaged((void**)&hostPageStatTable, numMallocManaged*sizeof(MemAccStatTable_t)));
    cudaPinned(hostPageStatTable, numMallocManaged*sizeof(MemAccStatTable_t));
    //cudaPinToGpu(hostPageStatTable, numMallocManaged*sizeof(MemAccStatTable_t));
    
    /* should write the table address to nvidia driver */
    uvm_fd = open("/dev/nvidia-uvm", O_RDWR); 
    if (uvm_fd < 0) {
        DEBUG_PRT("\nopen %s failed.\n\n", "/dev/nvidia-uvm");
        exit(-1);
    }
    
    /* Intrumentation enable flag */

    DEBUG_PRT("host page stat table top-level address %lx, num %u\n", (uint64_t)hostPageStatTable, (uint32_t)numMallocManaged);
    uvm_params _params = { (uint64_t)hostPageStatTable, (uint32_t)numMallocManaged };
    ioctl(uvm_fd, UVM_MAP_MEM_ACC_STAT_TABLE, &_params);

#endif

    psu_cuc1057_channel_device_init<<<1,1>>>((uint64_t)doorbell, 
                                             (uint64_t)dev_buff, 
                                             buff_size
                                             #if (NUM_BUFFERS > 0)
                                             , NUM_BUFFERS
                                             #endif
                                             #if StatInGPU
                                             ,(uint64_t)hostPageStatTable
                                             ,(uint32_t)numMallocManaged
                                             #endif
                                             );
    
    DEBUG_PRT("\t === init done === \n");
    
    #if RAW_TRANSFER
    recv_thread_started = true;
    pthread_create(&collector_thread, NULL, &channel_host_handle, NULL); 
    #endif

#endif
}

#define CHECK_ALL_DOORBELL()\
{\
    for (int _i = 0; _i < NUM_BUFFERS; _i++) {\
        while ((doorbell[_i]));\
    }\
}

__host__ void stat_table_debug_print()
{
    uint32_t page_num;
    for (int i = 0; i < stat_table_index; i++) {
        DEBUG_PRT("hostPageStatTable[%d]\n", i);
        page_num = (hostPageStatTable[i].region_size + (PAGE_SIZE-1))/PAGE_SIZE;
        if (hostPageStatTable[i].stat_table) {
            page_num = (hostPageStatTable[i].region_size + (PAGE_SIZE-1))/PAGE_SIZE;
            for (int j = 0; j < page_num*ENT_PER_PAGE; j++ ) {
                DEBUG_PRT("addr %lx : %d\n", (uint64_t)&(hostPageStatTable[i].stat_table[j]), hostPageStatTable[i].stat_table[j]);
            }
        }
    }
}

__host__ void channel_host_end()
{
#if INSTRUMENTED 
#if RAW_TRANSFER
    #if NUM_BUFFERS > 1
    CHECK_ALL_DOORBELL();
    #else
    while (*doorbell);
    #endif
#endif
    //pthread_join(collector_thread, NULL);
#if StatInGPU
    //channel_host.get_page_stat(); 
#endif
    //pthread_cancel(collector_thread);
    
    if (buff) {
        cudaFree(buff);
        DEBUG_PRT("Release Channel Buffer by Host\n");
    }
#if StatInGPU
    //stat_table_debug_print();
    for (int i = 0; i < stat_table_size; i++) {
        //if (mallocManagedArr[i].stat_table) {
        //    cudaFree(mallocManagedArr[i].stat_table);
        //}
        if (hostPageStatTable[i].stat_table) {
            cudaFree(hostPageStatTable[i].stat_table);
            DEBUG_PRT("free quant table %d\n", i);
        }
    }
    /*
    if (mallocManagedArr) {
        cudaFree(mallocManagedArr);
    }
    */
    if (hostPageStatTable) {
        cudaFree(hostPageStatTable);
        DEBUG_PRT("free stat table\n");
    }
    
    if (uvm_fd > 0) {
        close(uvm_fd);
    }
#endif

#endif 
}

/*****************************************************************************/
/*****************************************************************************/



