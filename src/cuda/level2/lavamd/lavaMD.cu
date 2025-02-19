////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\lavamd\lavaMD.cu
//
// summary:	Lava md class
// 
// origin: 	14 APR 2011 Lukasz G. Szafaryn, from Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false
#include "./util/num/num.h"				// (in path specified here)
#include "./lavaMD.h"						// (in the current directory)
#include "./kernel/kernel_gpu_cuda_wrapper.h"	// (in library path specified here)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only use demand paging");
    op.addOption("boxes1d", OPT_INT, "0",
            "specify number of boxes in single dimension, total box number is that^3");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

//void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op, ofstream &ofile, sem_t *sem) {
    printf("Running LavaMD\n");

    bool quiet = op.getOptionBool("quiet");

    // get boxes1d arg value
    int boxes1d = op.getOptionInt("boxes1d");
    if (boxes1d == 0) {
        int probSizes[4] = {1, 8, 64, 64};
        boxes1d = probSizes[op.getOptionInt("size") - 1];
    }

    if (!quiet) {
        printf("Thread block size of kernel = %d \n", NUMBER_THREADS);
        printf("Configuration used: boxes1d = %d\n", boxes1d);
    }

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++) {
        if (!quiet) { printf("Pass %d: ", i); }
        runTest(resultDB, op, boxes1d, ofile, sem);
        if (!quiet) { printf("Done.\n"); }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
/// <param name="boxes1d"> 	The boxes 1d. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void runTest(ResultDatabase &resultDB, OptionParser &op, int boxes1d, ofstream &ofile, 
        sem_t *sem) {
    bool uvm = op.getOptionBool("uvm");
    bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    bool copy = op.getOptionBool("copy");
    bool pageable = op.getOptionBool("pageable");
    const bool is_barrier = op.getOptionBool("sem");
    // random generator seed set to random value - time in this case
    srand(SEED);

    // counters
    int i, j, k, l, m, n;

    // system memory
    par_str par_cpu;
    dim_str dim_cpu;
    box_str* box_cpu;
    FOUR_VECTOR* rv_cpu;
    fp* qv_cpu;
    FOUR_VECTOR* fv_cpu;
    int nh;

    dim_cpu.boxes1d_arg = boxes1d;
    par_cpu.alpha = 0.5;

    // total number of boxes
    dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

    // how many particles space has in each direction
    dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
    dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
    dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

    // box array
    dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

    // allocate boxes
    if (uvm || uvm_prefetch) {
        checkCudaErrors(cudaMallocManaged(&box_cpu, dim_cpu.box_mem));
    } else if (copy) {
        checkCudaErrors(cudaMallocHost(&box_cpu, dim_cpu.box_mem));
        assert(box_cpu);
    } else {
        box_cpu = (box_str *)malloc(dim_cpu.box_mem);
        assert(box_cpu);
    }

    // initialize number of home boxes
    nh = 0;

    // home boxes in z direction
    for (i=0; i<dim_cpu.boxes1d_arg; i++) {
        // home boxes in y direction
        for (j=0; j<dim_cpu.boxes1d_arg; j++) {
            // home boxes in x direction
            for (k=0; k<dim_cpu.boxes1d_arg; k++) {

                // current home box
                box_cpu[nh].x = k;
                box_cpu[nh].y = j;
                box_cpu[nh].z = i;
                box_cpu[nh].number = nh;
                box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

                // initialize number of neighbor boxes
                box_cpu[nh].nn = 0;

                // neighbor boxes in z direction
                for(l=-1; l<2; l++){
                    // neighbor boxes in y direction
                    for(m=-1; m<2; m++){
                        // neighbor boxes in x direction
                        for(n=-1; n<2; n++){

                            // check if (this neighbor exists) and (it is not the same as home box)
                            if(		(((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)	&&
                                    (l==0 && m==0 && n==0)==false	){

                                // current neighbor box
                                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                                box_cpu[nh].nei[box_cpu[nh].nn].number =	(box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
                                    (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
                                    box_cpu[nh].nei[box_cpu[nh].nn].x;
                                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                                // increment neighbor box
                                box_cpu[nh].nn = box_cpu[nh].nn + 1;

                            }

                        } // neighbor boxes in x direction
                    } // neighbor boxes in y direction
                } // neighbor boxes in z direction

                // increment home box
                nh = nh + 1;

            } // home boxes in x direction
        } // home boxes in y direction
    } // home boxes in z direction

    // input (distances)
    if (uvm || uvm_prefetch) {
        checkCudaErrors(cudaMallocManaged(&rv_cpu, dim_cpu.space_mem));
    } else if (copy) {
        checkCudaErrors(cudaMallocHost(&rv_cpu, dim_cpu.space_mem));
        assert(rv_cpu);
    } else {
        rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
        assert(rv_cpu);
    }

    for (i=0; i<dim_cpu.space_elem; i=i+1) {
        rv_cpu[i].v = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
        rv_cpu[i].x = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
        rv_cpu[i].y = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
        rv_cpu[i].z = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
    }

    // input (charge)
    if (uvm || uvm_prefetch) {
        checkCudaErrors(cudaMallocManaged(&qv_cpu, dim_cpu.space_mem2));
    } else if (copy) {
        checkCudaErrors(cudaMallocHost(&qv_cpu, dim_cpu.space_mem2));
        assert(qv_cpu);
    } else {
        qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
        assert(qv_cpu);
    }

    for(i=0; i<dim_cpu.space_elem; i=i+1){
        qv_cpu[i] = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
    }

    // output (forces)
    if (uvm || uvm_prefetch) {
        checkCudaErrors(cudaMallocManaged(&fv_cpu, dim_cpu.space_mem));
    } else if (copy) {
        checkCudaErrors(cudaMallocHost(&fv_cpu, dim_cpu.space_mem));
        assert(fv_cpu);
    } else {
        fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
        assert(fv_cpu);
    }

    for (i=0; i<dim_cpu.space_elem; i=i+1) {
        fv_cpu[i].v = 0;								// set to 0, because kernels keeps adding to initial value
        fv_cpu[i].x = 0;								// set to 0, because kernels keeps adding to initial value
        fv_cpu[i].y = 0;								// set to 0, because kernels keeps adding to initial value
        fv_cpu[i].z = 0;								// set to 0, because kernels keeps adding to initial value
    }

    kernel_gpu_cuda_wrapper(par_cpu,
            dim_cpu,
            box_cpu,
            rv_cpu,
            qv_cpu,
            fv_cpu,
            resultDB,
            op,
            ofile, sem);

    string outfile = op.getOptionString("outputFile");
    if (outfile != "") {
        FILE *fptr;
        fptr = fopen("result.txt", "w");	
        for(i=0; i<dim_cpu.space_elem; i=i+1){
            fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
        }
        fclose(fptr);
    }

    if (uvm || uvm_prefetch) {
        checkCudaErrors(cudaFree(rv_cpu));
        checkCudaErrors(cudaFree(qv_cpu));
        checkCudaErrors(cudaFree(fv_cpu));
        checkCudaErrors(cudaFree(box_cpu));
    } else if (copy) {
        checkCudaErrors(cudaFreeHost(rv_cpu));
        checkCudaErrors(cudaFreeHost(qv_cpu));
        checkCudaErrors(cudaFreeHost(fv_cpu));
        checkCudaErrors(cudaFreeHost(box_cpu));
    } else {
        free(rv_cpu);
        free(qv_cpu);
        free(fv_cpu);
        free(box_cpu);
    }
}
