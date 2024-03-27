////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\where\where.cu
//
// summary:	Where class
// 
// origin: 
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
    
/// <summary>	The kernel time. </summary>
float kernelTime = 0.0f;
/// <summary>	The transfer time. </summary>
float transferTime = 0.0f;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the stop. </summary>
///
/// <value>	The stop. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

cudaEvent_t start, stop;
/// <summary>	The elapsed time. </summary>
float elapsedTime;

// (taeklim): Warp coalescing
#define WARP_SHIFT 5
#define WARP_SIZE 32

// 512 Bytes structure
#define ELEM_NUM 64
// We only consider 16 elements (128B) from this structure
#define TARGET_NUM 16
typedef struct data {
    uint64_t num[ELEM_NUM];
} data;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Checks. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="val">  	The value. </param>
/// <param name="bound">	The bound. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool check(uint64_t val, int bound) {
    return (val < bound);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Mark matches. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr">	  	[in,out] If non-null, the array. </param>
/// <param name="results">	[in,out] If non-null, the results. </param>
/// <param name="size">   	The size. </param>
/// <param name="bound">  	The bound. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void markMatches(data *data_arr, int *results, int size, int bound) {
    // Block index
    int bx = blockIdx.x;
    // Thread index
    int tx = threadIdx.x;

    int tid = (blockDim.x * bx) + tx;

    if (tid < size) {
        // We only match TARGET_NUMs
        for (int i = 0; i < TARGET_NUM; i++) {
            if (check(data_arr[tid].num[i], bound)) {
                results[tid * TARGET_NUM + i] = 1;
            } else {
                results[tid * TARGET_NUM + i] = 0;
            }
        }
    }
}


__global__ void markMatchesCoal(data *data_arr, int *results, int size, int bound) {
    // Block index
    int bx = blockIdx.x;
    // Thread index
    int tx = threadIdx.x;
    int tid = (blockDim.x * bx) + tx;
    int warp_id = tid >> WARP_SHIFT;
    int lane_id = tid & ((1 << WARP_SHIFT) - 1);

    if (warp_id < size) {
        if (lane_id < TARGET_NUM) {
            if (check(data_arr[warp_id].num[lane_id], bound)) {
                results[warp_id * TARGET_NUM + lane_id] = 1;
            } else {
                results[warp_id * TARGET_NUM + lane_id] = 0;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Map matches. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr">	  	[in,out] If non-null, the array. </param>
/// <param name="results">	[in,out] If non-null, the results. </param>
/// <param name="prefix"> 	[in,out] If non-null, the prefix. </param>
/// <param name="final">  	[in,out] If non-null, the final. </param>
/// <param name="size">   	The size. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mapMatches(data *data_arr, int *results, int *prefix, int *final, int size) {
    // Block index
    int bx = blockIdx.x;
    // Thread index
    int tx = threadIdx.x;
    int tid = (blockDim.x * bx) + tx;

    //for( ; tid < size; tid += blockDim.x * gridDim.x) {
    if (tid < size) {
        for (int i = 0; i < TARGET_NUM; i++) {
            if(results[tid * TARGET_NUM + i]) {
                final[prefix[tid * TARGET_NUM + i]] = data_arr[tid].num[i];
            }
        }
    }
}

__global__ void mapMatchesCoal(data *data_arr, int *results, int *prefix, int *final, int size) {
    // Block index
    int bx = blockIdx.x;
    // Thread index
    int tx = threadIdx.x;
    int tid = (blockDim.x * bx) + tx;
    int warp_id = tid >> WARP_SHIFT;
    int lane_id = tid & ((1 << WARP_SHIFT) - 1);

    if (warp_id < size) {
        if (lane_id < TARGET_NUM) {
            if(results[warp_id * TARGET_NUM + lane_id]) {
                final[prefix[warp_id * TARGET_NUM + lane_id]] = data_arr[warp_id].num[lane_id];
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Seed array. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr"> 	[in,out] If non-null, the array. </param>
/// <param name="size">	The size. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void seedArr(int *arr, int size) {
    for(int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

void seedDataArr(data *data_arr, int size) {
    for(int i = 0; i < size; i++) {
        uint64_t rand_num = rand() % 100;
        //for (int j = 0; j < ELEM_NUM; j++)
        for (int j = 0; j < ELEM_NUM; j++)
            data_arr[i].num[j] = rand_num;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Wheres. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="size">	   	The size. </param>
/// <param name="coverage">	The coverage. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////


void where(ResultDatabase &resultDB, OptionParser &op, int size, int coverage, ofstream &ofile, sem_t *sem) {
    const bool uvm = op.getOptionBool("uvm");
    const bool zero_copy = op.getOptionBool("zero-copy");
    const bool copy = op.getOptionBool("copy");
    const bool pageable = op.getOptionBool("pageable");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    const bool dha = op.getOptionBool("dha");
    const bool coal = op.getOptionBool("coal");
    string bench_name = op.getOptionString("bench");
    const bool is_barrier = op.getOptionBool("sem");
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));

    int gpu_num = 2;

    data *data_arr;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise || zero_copy) {
        checkCudaErrors(cudaMallocManaged(&data_arr, sizeof(data) * size));
    } else if (copy) {
        checkCudaErrors(cudaMallocHost(&data_arr, sizeof(data) * size));
    } else if (dha) {
        checkCudaErrors(cudaHostAlloc(&data_arr, sizeof(data) * size, cudaHostAllocDefault));
    } else if (pageable) {
        data_arr = (data*)malloc(sizeof(data) * size);
        assert(data_arr);
    }
    seedDataArr(data_arr, size);
    printf("Done with initializing struct members\n");

    int *final;
    int *d_results;
    int *d_prefix;
    int *d_final;
    data *d_data_arr;
    
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise || zero_copy) {
        d_data_arr = data_arr;
        checkCudaErrors(cudaMallocManaged((void**) &d_results, sizeof(int) * size * TARGET_NUM));
        checkCudaErrors(cudaMallocManaged((void**) &d_prefix, sizeof(int) * size * TARGET_NUM));
    } else if (copy || pageable) {
        checkCudaErrors(cudaMalloc((void**) &d_data_arr, sizeof(data) * size));
        checkCudaErrors(cudaMalloc((void**) &d_results, sizeof(int) * size));
        checkCudaErrors(cudaMalloc((void**) &d_prefix, sizeof(int) * size));
    } else if (dha) {
        d_data_arr = data_arr;
        checkCudaErrors(cudaHostAlloc(&d_results, sizeof(int) * size, cudaHostAllocDefault));
        checkCudaErrors(cudaHostAlloc(&d_prefix, sizeof(int) * size, cudaHostAllocDefault));
    }

    if (uvm || dha) {
        checkCudaErrors(cudaEventRecord(start, 0));
        // do nothing
    } else if (zero_copy) {
        checkCudaErrors(cudaEventRecord(start, 0));
        checkCudaErrors(cudaMemAdvise(d_data_arr, sizeof(data) * size, cudaMemAdviseSetAccessedBy, 0));
    } else if (uvm_advise) {
        checkCudaErrors(cudaEventRecord(start, 0));
        checkCudaErrors(cudaMemAdvise(d_data_arr, sizeof(data) * size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(d_data_arr, sizeof(data) * size, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaEventRecord(start, 0));
        checkCudaErrors(cudaMemPrefetchAsync(d_data_arr, sizeof(data) * size, device));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaEventRecord(start, 0));
        checkCudaErrors(cudaMemAdvise(d_data_arr, sizeof(data) * size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(d_data_arr, sizeof(data) * size, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(d_data_arr, sizeof(data) * size, device));
    } else if (copy || pageable) {
        if (is_barrier && pageable) {
            int sval;
            sem_post(sem);
            sem_getvalue(sem, &sval);
            while (sval == 1) {
                sem_getvalue(sem, &sval);
            }
            printf("[Barrier] Copying starts\n");
        }
        checkCudaErrors(cudaEventRecord(start, 0));
        checkCudaErrors(cudaMemcpy(d_data_arr, data_arr, sizeof(int) * size, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;

    // (taeklim): Waiting for the other apps finishes the initialization
    if (is_barrier && uvm) {
        int sval;
        sem_post(sem);
        sem_getvalue(sem, &sval);
        while (sval == 1) {
            sem_getvalue(sem, &sval);
        }
        printf("[Barrier] Kernel starts\n");
    }
    //dim3 grid(size / 1024 + 1, 1, 1);
    dim3 grid((size * WARP_SIZE + 1024) / 1024, 1, 1);
    dim3 threads(1024, 1, 1);
    checkCudaErrors(cudaEventRecord(start, 0));
    if (coal)
        markMatchesCoal<<<grid, threads>>>(d_data_arr, d_results, size, coverage);
    else  
        markMatches<<<grid, threads>>>(d_data_arr, d_results, size, coverage);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    int temp_size = 0;
    for (int i = 0; i < size; i++) {
        if (d_results[i] == 1) {
            temp_size++;
        }
    }
    printf("temp_size:%d\n", temp_size);

    checkCudaErrors(cudaEventRecord(start, 0));
    thrust::exclusive_scan(thrust::device, d_results, d_results + size, d_prefix);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    int matchSize;
    checkCudaErrors(cudaEventRecord(start, 0));
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise || zero_copy) {
        matchSize = (int)*(d_prefix + size - 1);
    } else {
        checkCudaErrors(cudaMemcpy(&matchSize, d_prefix + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;
    matchSize++;
    cout << "matchsize: " << matchSize << endl;


    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise || zero_copy) {
        checkCudaErrors(cudaMallocManaged( (void**) &d_final, sizeof(int) * matchSize));
        final = d_final;
    } else {
        checkCudaErrors(cudaMalloc( (void**) &d_final, sizeof(int) * matchSize));
        final = (int*)malloc(sizeof(int) * matchSize);
        assert(final);
    }

    checkCudaErrors(cudaEventRecord(start, 0));
    if (coal)
        mapMatchesCoal<<<grid, threads>>>(d_data_arr, d_results, d_prefix, d_final, size);
    else 
        mapMatches<<<grid, threads>>>(d_data_arr, d_results, d_prefix, d_final, size);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    checkCudaErrors(cudaEventRecord(start, 0));
    // No cpy just demand paging
    if (uvm) {
        // Do nothing
    } else if (zero_copy) {
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetAccessedBy, 0));
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(final, sizeof(int) * matchSize, cudaCpuDeviceId));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(final, sizeof(int) * matchSize, cudaCpuDeviceId));
    } else if (copy || pageable) {
        checkCudaErrors(cudaMemcpy(final, d_final, sizeof(int) * matchSize, cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise || zero_copy) {
        checkCudaErrors(cudaFree(d_data_arr));
        checkCudaErrors(cudaFree(d_results));
        checkCudaErrors(cudaFree(d_prefix));
        checkCudaErrors(cudaFree(d_final));
    } else if (pageable) {
        free(data_arr);
        free(final);
        checkCudaErrors(cudaFree(d_data_arr));
        checkCudaErrors(cudaFree(d_results));
        checkCudaErrors(cudaFree(d_prefix));
        checkCudaErrors(cudaFree(d_final));
    } else if (copy) {
        checkCudaErrors(cudaFreeHost(data_arr));
        free(final);

        checkCudaErrors(cudaFree(d_data_arr));
        checkCudaErrors(cudaFree(d_results));
        checkCudaErrors(cudaFree(d_prefix));
        checkCudaErrors(cudaFree(d_final));
    } else if (dha) {
        checkCudaErrors(cudaFreeHost(data_arr));
        checkCudaErrors(cudaFreeHost(d_results));
        checkCudaErrors(cudaFreeHost(d_prefix));
        free(final);

    }
    
    char atts[1024];
    sprintf(atts, "size:%d, coverage:%d", size, coverage);
    resultDB.AddResult("where_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("where_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("where_total_time", atts, "sec", kernelTime+transferTime);
    resultDB.AddResult("where_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
    ofile << bench_name << ", " << kernelTime + transferTime << ", " << endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("length", OPT_INT, "0", "number of elements in input");
  op.addOption("coverage", OPT_INT, "-1", "0 to 100 percentage of elements to allow through where filter");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

//void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op, ofstream &ofile, sem_t *sem) {
    printf("Running Where\n");

    srand(7);

    bool quiet = op.getOptionBool("quiet");
    int size = op.getOptionInt("length");
    int coverage = op.getOptionInt("coverage");
    if (size == 0 || coverage == -1) {
        //int sizes[5] = {1000, 10000, 500000000, 1000000000, 1050000000};
        //int sizes[5] = {1000, 10000, 500000000, 100000000, 1050000000};
        int sizes[5] = {1000, 10000, 500000000, 25000000, 1050000000};
        int coverages[5] = {20, 30, 40, 80, 240};
        size = sizes[op.getOptionInt("size") - 1];
        coverage = coverages[op.getOptionInt("size") - 1];
        coverage = 20;
    }

    if (!quiet) {
        printf("Using size=%d, coverage=%d\n", size, coverage);
    }

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++) {
        kernelTime = 0.0f;
        transferTime = 0.0f;
        if(!quiet) {
            printf("Pass %d: ", i);
        }
        where(resultDB, op, size, coverage, ofile, sem);
        if(!quiet) {
            printf("Done.\n");
        }
    }
}
