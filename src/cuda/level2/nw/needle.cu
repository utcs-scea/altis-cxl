////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\nw\needle.cu
//
// summary:	Needle class
// 
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>

#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "needle.h"
#include "needle_kernel.cu"

#define LIMIT -999
#define SEED 7

int max_rows, max_cols, penalty;

void runTest(ResultDatabase &resultDB, OptionParser &op, ofstream &ofile, sem_t *sem);

/// <summary>	The blosum 62[24][24]. </summary>
int blosum62[24][24] = {{4,  -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1,
                         -1, -2, -1, 1,  0, -3, -2, 0, -2, -1, 0,  -4},
                        {-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,
                         -1, -3, -2, -1, -1, -3, -2, -3, -1, 0,  -1, -4},
                        {-2, 0,  6,  1, -3, 0,  0,  0,  1, -3, -3, 0,
                         -2, -3, -2, 1, 0,  -4, -2, -3, 3, 0,  -1, -4},
                        {-2, -2, 1,  6, -3, 0,  2,  -1, -1, -3, -4, -1,
                         -3, -3, -1, 0, -1, -4, -3, -3, 4,  1,  -1, -4},
                        {0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3,
                         -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
                        {-1, 1,  0,  0, -3, 5,  2,  -2, 0, -3, -2, 1,
                         0,  -3, -1, 0, -1, -2, -1, -2, 0, 3,  -1, -4},
                        {-1, 0,  0,  2, -4, 2,  5,  -2, 0, -3, -3, 1,
                         -2, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2,
                         -3, -3, -2, 0,  -2, -2, -3, -3, -1, -2, -1, -4},
                        {-2, 0,  1,  -1, -3, 0,  0, -2, 8, -3, -3, -1,
                         -2, -1, -2, -1, -2, -2, 2, -3, 0, 0,  -1, -4},
                        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3,
                         1,  0,  -3, -2, -1, -3, -1, 3,  -3, -3, -1, -4},
                        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2,
                         2,  0,  -3, -2, -1, -2, -1, 1,  -4, -3, -1, -4},
                        {-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,
                         -1, -3, -1, 0,  -1, -3, -2, -2, 0,  1,  -1, -4},
                        {-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1,
                         5,  0,  -2, -1, -1, -1, -1, 1,  -3, -1, -1, -4},
                        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3,
                         0,  6,  -4, -2, -2, 1,  3,  -1, -3, -3, -1, -4},
                        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1,
                         -2, -4, 7,  -1, -1, -4, -3, -2, -2, -1, -2, -4},
                        {1,  -1, 1,  0, -1, 0,  0,  0,  -1, -2, -2, 0,
                         -1, -2, -1, 4, 1,  -3, -2, -2, 0,  0,  0,  -4},
                        {0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1,
                         -1, -2, -1, 1,  5,  -2, -2, 0,  -1, -1, 0,  -4},
                        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3,
                         -1, 1,  -4, -3, -2, 11, 2,  -3, -4, -3, -2, -4},
                        {-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2,
                         -1, 3,  -3, -2, -2, 2,  7,  -1, -3, -2, -1, -4},
                        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2,
                         1, -1, -2, -2, 0,  -3, -1, 4,  -3, -2, -1, -4},
                        {-2, -1, 3,  4, -3, 0,  1,  -1, 0, -3, -4, 0,
                         -3, -3, -2, 0, -1, -4, -3, -3, 4, 1,  -1, -4},
                        {-1, 0,  0,  1, -3, 3,  4,  -2, 0, -3, -3, 1,
                         -1, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -2, 0,  0,  -2, -1, -1, -1, -1, -1, -4},
                        {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                         -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Add benchmark specific options parsing.  The user is allowed to specify
/// the size of the input data in kiB.. </summary>
///
/// <remarks>	Ed, 5/20/2020.
/// 			Anthony Danalis, 9/08, 2009
///
/// <param name="op">	[in,out] the options parser / parameter database. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only demand paging");
  op.addOption("dimensions", OPT_INT, "0", "dimensions");
  op.addOption("penalty", OPT_INT, "10", "penalty");
  op.addOption("resultsfile", OPT_STRING, "", "file to write results to");
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020.
/// 			Kyle Spafford, 8/13/2009 </remarks>
///
/// <param name="resultDB">	[in,out] results from the benchmark are stored in this db. </param>
/// <param name="op">	   	[in,out] the options parser / parameter database. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

//void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op, ofstream &ofile, sem_t *sem) {
  printf("Running Needleman-Wunsch\n");

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  bool quiet = op.getOptionBool("quiet");
  int dim = op.getOptionInt("dimensions");
  penalty = op.getOptionInt("penalty");

  if(penalty < 0) {
      printf("Error: Penalty must be a positive number.\n");
      return;
  }
  if(dim < 0) {
      printf("Error: Dimensions must be positive.\n");
      return;
  }

  if (dim == 0) {
    int problemSizes[4] = {100, 1000, 6000, 40000};
    dim = problemSizes[op.getOptionInt("size") - 1];
  }

  long long num_items = (long long)dim * (long long)dim;
  if(num_items >= INT_MAX) {
      printf("Error: Total size cannot exceed INT_MAX");
      return;
  }

  if(!quiet) {
      printf("WG size of kernel = %d \n", BLOCK_SIZE);
      printf("Max Rows x Cols: %dx%d\n", dim, dim);
      printf("Penalty: %d\n\n", penalty);
  }
  srand(SEED);

  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
      if(!quiet) {
          printf("Pass %d: ", i);
      }
      max_rows = dim;
      max_cols = dim;
      runTest(resultDB, op, ofile, sem);
      if(!quiet) {
          printf("Done.\n");
      }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void runTest(ResultDatabase &resultDB, OptionParser &op, ofstream &ofile, sem_t *sem) {
  bool uvm = op.getOptionBool("uvm");
  bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  bool copy = op.getOptionBool("copy");
  bool quiet = op.getOptionBool("quiet");
  bool pageable = op.getOptionBool("pageable");
  int *input_itemsets, *output_itemsets, *referrence;
  int *matrix_cuda, *referrence_cuda;
  int size;
  int device = 0;
  checkCudaErrors(cudaGetDevice(&device));
  const bool is_barrier = op.getOptionBool("sem");
  string bench_name = op.getOptionString("bench");

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  
  if (uvm || uvm_prefetch) {
    checkCudaErrors(cudaMallocManaged(&referrence, max_rows * max_cols * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&input_itemsets, max_rows * max_cols * sizeof(int)));
  } else if (copy) {
    checkCudaErrors(cudaMallocHost(&referrence, max_rows * max_cols * sizeof(int)));
    //checkCudaErrors(cudaHostAlloc(&referrence, max_rows * max_cols * sizeof(int), cudaHostAllocDefault));
    assert(referrence);
    checkCudaErrors(cudaMallocHost(&input_itemsets, max_rows * max_cols * sizeof(int)));
    assert(input_itemsets);
    checkCudaErrors(cudaMallocHost(&output_itemsets, max_rows * max_cols * sizeof(int)));
    assert(output_itemsets);
  } else if (pageable) {
    referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
    assert(referrence);
    input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
    assert(input_itemsets);
    output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
    assert(output_itemsets);
  }

  if (!input_itemsets) {
      fprintf(stderr, "Error: Can not allocate memory\n");
      exit(0);
  }

  for (int i = 0; i < max_cols; i++) {
    for (int j = 0; j < max_rows; j++) {
      input_itemsets[i * max_cols + j] = 0;
    }
  }

  for (int i = 1; i < max_rows; i++) {  // please define your own sequence.
    input_itemsets[i * max_cols] = rand() % 10 + 1;
  }
  for (int j = 1; j < max_cols; j++) {  // please define your own sequence.
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (int i = 1; i < max_cols; i++) {
    for (int j = 1; j < max_rows; j++) {
      referrence[i * max_cols + j] =
          blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
    }
  }

  for (int i = 1; i < max_rows; i++)
    input_itemsets[i * max_cols] = -i * penalty;
  for (int j = 1; j < max_cols; j++) input_itemsets[j] = -j * penalty;

  size = max_cols * max_rows;

  if (uvm || uvm_prefetch) {
    // Do nothing
  } else if (copy || pageable) {
    checkCudaErrors(cudaMalloc((void **)&referrence_cuda, sizeof(int) * size));
    checkCudaErrors(cudaMalloc((void **)&matrix_cuda, sizeof(int) * size));
  } 

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  double transferTime = 0.;
  double kernelTime = 0;
  double totalTime = 0;

  // Notice that here we used demand paging so no cpy time included, could also use HyperQ
  if (uvm) {
      cudaEventRecord(start, 0);
    referrence_cuda = referrence;
    matrix_cuda = input_itemsets;
  } else if (uvm_prefetch) {
      cudaEventRecord(start, 0);
    referrence_cuda = referrence;
    matrix_cuda = input_itemsets;
      checkCudaErrors(cudaMemPrefetchAsync(referrence_cuda, sizeof(int) * size , device));
      cudaStream_t s1;
      checkCudaErrors(cudaStreamCreate(&s1));
      checkCudaErrors(cudaMemPrefetchAsync(matrix_cuda, sizeof(int) * size , device, s1));
      checkCudaErrors(cudaStreamDestroy(s1));
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
    cudaEventRecord(start, 0);
    checkCudaErrors(cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size,
            cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size,
            cudaMemcpyHostToDevice));
  } 
  printf("Done copying...\n");
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds

  dim3 dimGrid;
  dim3 dimBlock(BLOCK_SIZE, 1);
  int block_width = (max_cols - 1) / BLOCK_SIZE;

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
  cudaEventRecord(start, 0);
  // process top-left matrix
  for (int i = 1; i <= block_width; i++) {
    dimGrid.x = i;
    dimGrid.y = 1;
    //cudaEventRecord(start, 0);
    needle_cuda_shared_1<<<dimGrid, dimBlock>>>(
            referrence_cuda, matrix_cuda, max_cols, penalty, i, block_width);
    //cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&elapsedTime, start, stop);
//    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();
  }
  // process bottom-right matrix
  for (int i = block_width - 1; i >= 1; i--) {
    dimGrid.x = i;
    dimGrid.y = 1;
//    cudaEventRecord(start, 0);
    needle_cuda_shared_2<<<dimGrid, dimBlock>>>(
        referrence_cuda, matrix_cuda, max_cols, penalty, i, block_width);
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&elapsedTime, start, stop);
//    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  kernelTime += elapsedTime * 1.e-3;

  printf("Done kernel...\n");

  cudaEventRecord(start, 0);
  if (uvm || uvm_prefetch) {
    output_itemsets = matrix_cuda;
    checkCudaErrors(cudaMemPrefetchAsync(output_itemsets, sizeof(int) * size, cudaCpuDeviceId));
    checkCudaErrors(cudaStreamSynchronize(0));
  } else if (copy || pageable) {
    checkCudaErrors(cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size,
            cudaMemcpyDeviceToHost));
  }
  printf("Done copying...\n");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds
  //kernelTime += elapsedTime * 1.e-3;
  totalTime += kernelTime + transferTime;

//  string outfile = op.getOptionString("outputFile");
//  if (outfile != "") {
////      FILE *fpo = fopen(outfile.c_str(), "w");
////      if(!quiet) {
////        fprintf(fpo, "Print traceback value GPU to %s:\n", outfile.c_str());
////      }
//
//      for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;) {
//          int nw, n, w, traceback;
//          if (i == max_rows - 2 && j == max_rows - 2) {
//              // print the first element
//              fprintf(fpo, "%d ", output_itemsets[i*max_cols+j]);
//          }
//          if (i == 0 && j == 0) {
//              break;
//          }
//          if (i > 0 && j > 0) {
//              nw = output_itemsets[(i - 1) * max_cols + j - 1];
//              w = output_itemsets[i * max_cols + j - 1];
//              n = output_itemsets[(i - 1) * max_cols + j];
//          } else if (i == 0) {
//              nw = n = LIMIT;
//              w = output_itemsets[i * max_cols + j - 1];
//          } else if (j == 0) {
//              nw = w = LIMIT;
//              n = output_itemsets[(i - 1) * max_cols + j];
//          } else {
//          }
//
//          // traceback = maximum(nw, w, n);
//          int new_nw, new_w, new_n;
//          new_nw = nw + referrence[i * max_cols + j];
//          new_w = w - penalty;
//          new_n = n - penalty;
//
//          traceback = maximum(new_nw, new_w, new_n);
//          if (traceback == new_nw) {
//              traceback = nw;
//          }
//          if (traceback == new_w) {
//              traceback = w;
//          }
//          if (traceback == new_n) {
//              traceback = n;
//          }
//
//          fprintf(fpo, "%d ", traceback);
//          if (traceback == nw) {
//              i--;
//              j--;
//              continue;
//          } else if (traceback == w) {
//              j--;
//              continue;
//          } else if (traceback == n) {
//              i--;
//              continue;
//          } else {
//          }
//      }
//      fclose(fpo);
//  }
  printf("Done output...\n");

  // Cleanup memory
  if (uvm || uvm_prefetch) {
    checkCudaErrors(cudaFree(referrence_cuda));
    checkCudaErrors(cudaFree(matrix_cuda));
  } else if (copy) {
    checkCudaErrors(cudaFree(referrence_cuda));
    checkCudaErrors(cudaFree(matrix_cuda));
    checkCudaErrors(cudaFreeHost(output_itemsets));
    checkCudaErrors(cudaFreeHost(referrence));
    checkCudaErrors(cudaFreeHost(input_itemsets));
  } else {
    checkCudaErrors(cudaFree(referrence_cuda));
    checkCudaErrors(cudaFree(matrix_cuda));
    free(referrence);
    free(input_itemsets);
    free(output_itemsets);
  }

  char tmp[32];
  sprintf(tmp, "%ditems", size);
  string atts = string(tmp);
  //resultDB.AddResult("NW-TransferTime", atts, "sec", transferTime);
  resultDB.AddResult("NW-KernelTime", atts, "sec", kernelTime);
  resultDB.AddResult("NW-TotalTime", atts, "sec", totalTime);
  ofile << bench_name << ", " << totalTime << ", " << endl;
//  resultDB.AddResult("NW-TotalTime", atts, "sec", transferTime + kernelTime);
//  resultDB.AddResult("NW-Rate_Parity", atts, "N", transferTime / kernelTime);
//  resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
}
