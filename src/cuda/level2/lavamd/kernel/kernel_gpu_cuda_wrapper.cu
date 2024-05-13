//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../lavaMD.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer
#include "cudacommon.h"

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper.h"				// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda.cu"						// (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

#define MAX_STREAM 32

/// <summary>	An enum constant representing the void option. </summary>
void 
kernel_gpu_cuda_wrapper(par_str par_cpu,
						dim_str dim_cpu,
						box_str* box_cpu,
						FOUR_VECTOR* rv_cpu,
						fp* qv_cpu,
						FOUR_VECTOR* fv_cpu,
                        ResultDatabase &resultDB,
						OptionParser &op,
                        ofstream &ofile,
                        sem_t *sem)
{
	bool uvm = op.getOptionBool("uvm");
	bool zero_copy = op.getOptionBool("zero-copy");
	bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
	bool copy = op.getOptionBool("copy");
	bool pageable = op.getOptionBool("pageable");
	bool async = op.getOptionBool("async");
	bool pud = op.getOptionBool("pud");
    const bool is_barrier = op.getOptionBool("sem");
    string bench_name = op.getOptionString("bench");

    float kernelTime = 0.0f;
    float transferTime = 0.0f;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsedTime;
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));

    int s_id = 0;
    cudaStream_t streams[MAX_STREAM];
    if (async) {
        for (int s = 0; s < MAX_STREAM; s++) {
            cudaStreamCreate(&streams[s]);
        }
    }

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	checkCudaErrors(cudaDeviceSynchronize());

	//====================================================================================================100
	//	VARIABLES
	//====================================================================================================100

	box_str* d_box_gpu;
	FOUR_VECTOR* d_rv_gpu;
	fp* d_qv_gpu;
	FOUR_VECTOR* d_fv_gpu;

	dim3 threads;
	dim3 blocks;

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;

	//======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

	if (uvm || uvm_prefetch || zero_copy || pud) {
		d_box_gpu = box_cpu;
	} else if (copy) {
		checkCudaErrors(cudaMalloc(	(void **)&d_box_gpu,
					dim_cpu.box_mem));
	} else {
		checkCudaErrors(cudaMalloc(	(void **)&d_box_gpu,
					dim_cpu.box_mem));
    }

	//==================================================50
	//	rv
	//==================================================50

	if (uvm || uvm_prefetch || zero_copy || pud) {
		d_rv_gpu = rv_cpu;
	} else if (copy) {
		checkCudaErrors(cudaMalloc(	(void **)&d_rv_gpu, 
					dim_cpu.space_mem));
	} else {
		checkCudaErrors(cudaMalloc(	(void **)&d_rv_gpu, 
					dim_cpu.space_mem));
    }

	//==================================================50
	//	qv
	//==================================================50

	if (uvm || uvm_prefetch || zero_copy || pud) {
		d_qv_gpu = qv_cpu;
	} else if (copy) {
		checkCudaErrors(cudaMalloc(	(void **)&d_qv_gpu,
					dim_cpu.space_mem2));
	} else {
		checkCudaErrors(cudaMalloc(	(void **)&d_qv_gpu,
					dim_cpu.space_mem2));
    }

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	if (uvm || uvm_prefetch || zero_copy || pud) {
		d_fv_gpu = fv_cpu;
	} else if (copy) {
		checkCudaErrors(cudaMalloc(	(void **)&d_fv_gpu, 
					dim_cpu.space_mem));
	} else {
		checkCudaErrors(cudaMalloc(	(void **)&d_fv_gpu, 
					dim_cpu.space_mem));
    }

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

    if (!pageable && !copy)
        checkCudaErrors(cudaEventRecord(start, 0));

	if (uvm) {
		// Demand paging
	} else if (pud) {
        //checkCudaErrors(cudaMemAdvise(d_box_gpu, dim_cpu.box_mem, cudaMemAdviseSetReadMostly, 0));
    } else if (zero_copy) {
        checkCudaErrors(cudaMemAdvise(d_box_gpu, dim_cpu.box_mem, cudaMemAdviseSetAccessedBy, 0));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(d_box_gpu, dim_cpu.box_mem, device));
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
        if (async) {
            checkCudaErrors(cudaMemcpyAsync(d_box_gpu, 
                        box_cpu,
                        dim_cpu.box_mem, 
                        cudaMemcpyHostToDevice, streams[s_id++]));
        } else {
            checkCudaErrors(cudaMemcpy(	d_box_gpu, 
                        box_cpu,
                        dim_cpu.box_mem, 
                        cudaMemcpyHostToDevice));
        }
	} 

	//==================================================50
	//	rv
	//==================================================50
	
	if (uvm) {
		// Demand paging
	} else if (zero_copy || pud) {
        //checkCudaErrors(cudaMemAdvise(d_rv_gpu, dim_cpu.space_mem, cudaMemAdviseSetAccessedBy, 0));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(d_rv_gpu, dim_cpu.space_mem, device));
    } else if (copy || pageable) {
        if (async) {
            checkCudaErrors(cudaMemcpyAsync(d_rv_gpu,
                        rv_cpu,
                        dim_cpu.space_mem,
                        cudaMemcpyHostToDevice, streams[s_id++]));
        } else {
            checkCudaErrors(cudaMemcpy(	d_rv_gpu,
                        rv_cpu,
                        dim_cpu.space_mem,
                        cudaMemcpyHostToDevice));

        }
	} 

	//==================================================50
	//	qv
	//==================================================50

	if (uvm) {
		// Demand paging
	} else if (zero_copy || pud) {
        //checkCudaErrors(cudaMemAdvise(d_qv_gpu, dim_cpu.space_mem2, cudaMemAdviseSetAccessedBy, device));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(d_qv_gpu, dim_cpu.space_mem2, device));
    } else if (copy || pageable) {
        if (async) {
            checkCudaErrors(cudaMemcpyAsync(d_qv_gpu,
                        qv_cpu,
                        dim_cpu.space_mem2,
                        cudaMemcpyHostToDevice, streams[s_id++]));
        } else {
            checkCudaErrors(cudaMemcpy(	d_qv_gpu,
                        qv_cpu,
                        dim_cpu.space_mem2,
                        cudaMemcpyHostToDevice));

        }
	} 
	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	if (uvm) {
		// Demand paging
	} else if (zero_copy) {
        checkCudaErrors(cudaMemAdvise(d_fv_gpu, dim_cpu.space_mem, cudaMemAdviseSetAccessedBy, 0));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(d_fv_gpu, dim_cpu.space_mem, device));
    } else if (copy || pageable) {
        if (async) {
            checkCudaErrors(cudaMemcpyAsync(d_fv_gpu, 
                        fv_cpu, 
                        dim_cpu.space_mem, 
                        cudaMemcpyHostToDevice, streams[s_id++]));
        } else {
            checkCudaErrors(cudaMemcpy(	d_fv_gpu, 
                        fv_cpu, 
                        dim_cpu.space_mem, 
                        cudaMemcpyHostToDevice));
        }
	} 
	checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

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

	// launch kernel - all boxes
    checkCudaErrors(cudaEventRecord(start, 0));
	kernel_gpu_cuda<<<blocks, threads>>>(	par_cpu,
											dim_cpu,
											d_box_gpu,
											d_rv_gpu,
											d_qv_gpu,
											d_fv_gpu);
	checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;

    CHECK_CUDA_ERROR();
	checkCudaErrors(cudaDeviceSynchronize());

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)kernel
	//======================================================================================================================================================150

    checkCudaErrors(cudaEventRecord(start, 0));

	if (uvm || uvm_prefetch || zero_copy) {
		checkCudaErrors(cudaMemPrefetchAsync(d_fv_gpu, dim_cpu.space_mem, cudaCpuDeviceId));
        checkCudaErrors(cudaStreamSynchronize(0));
	} else if (copy || pageable) {
        if (async) {
            checkCudaErrors(cudaMemcpyAsync(fv_cpu, 
                        d_fv_gpu,
                        dim_cpu.space_mem, 
                        cudaMemcpyDeviceToHost, streams[s_id++]));
        } else {
            checkCudaErrors(cudaMemcpy(	fv_cpu, 
                        d_fv_gpu,
                        dim_cpu.space_mem, 
                        cudaMemcpyDeviceToHost));
        }
	} 
	checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;

    char atts[1024];
    sprintf(atts, "boxes1d:%d", dim_cpu.boxes1d_arg);
    resultDB.AddResult("lavamd_kernel_time", atts, "sec", kernelTime);
    //resultDB.AddResult("lavamd_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("lavamd_total_time", atts, "sec", kernelTime + transferTime);
    //resultDB.AddResult("lavamd_parity", atts, "N", transferTime / kernelTime);
    ofile << bench_name << ", " << kernelTime + transferTime << ", " << endl;

    if (async) {
        for (int s = 0; s < MAX_STREAM; s++) {
            cudaStreamDestroy(streams[s]);
        }
    }

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

	if (uvm) {
		// Demand paging, no need to free
	} else if (uvm_prefetch || zero_copy || pud) {

	} else if (copy) {
		checkCudaErrors(cudaFree(d_rv_gpu));
		checkCudaErrors(cudaFree(d_qv_gpu));
		checkCudaErrors(cudaFree(d_fv_gpu));
		checkCudaErrors(cudaFree(d_box_gpu));
	} else {
		checkCudaErrors(cudaFree(d_rv_gpu));
		checkCudaErrors(cudaFree(d_qv_gpu));
		checkCudaErrors(cudaFree(d_fv_gpu));
		checkCudaErrors(cudaFree(d_box_gpu));
    }
}
