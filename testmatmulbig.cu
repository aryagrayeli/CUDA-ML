#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "matmul.cu"
#include "matmul.h"
#include <sys/time.h>

int main() {

  srand(time(NULL));
  
  int power = 15;
  int bigN = (1<<power), bigM = (1<<15); 
  double *bigA_host, *bigB_host, *bigOutput_host;

  bigA_host = (double*) malloc(sizeof(double) * bigN * bigM);
  bigB_host = (double*) malloc(sizeof(double) * bigM);
  bigOutput_host = (double*) malloc(sizeof(double) * bigN);

  for(int i=0;i<bigN*bigM;i++) bigA_host[i] = ((double) rand())/RAND_MAX;
  for(int i=0;i<bigM;i++) bigB_host[i] = ((double) rand())/RAND_MAX;
  
  double *bigA, *bigB, *bigOutput;

  cudaMalloc(&bigA, bigN * bigM * sizeof(double));
  cudaMalloc(&bigB, bigM * sizeof(double));
  cudaMalloc(&bigOutput, bigN * sizeof(double));

  cudaMemcpy(bigA, bigA_host, bigN * bigM * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(bigB, bigB_host, bigM * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(bigOutput, bigOutput_host, bigN * sizeof(double), cudaMemcpyHostToDevice);

  printf("Test Big Matrix Multiplication\n");
  int threads = 1024;
  dim3 bigGridSize((bigN + threads - 1)/(threads), 1, 1);
  dim3 bigBlockSize(threads, 1, 1);

  time_t start_t, end_t;
  double diff_t;
  time(&start_t);
  struct timeval stop, start;
  gettimeofday(&start, NULL);
  matrix_mul<<<bigGridSize, bigBlockSize>>>(bigA, bigB, bigOutput, bigN, bigM);
  cudaDeviceSynchronize();
  gettimeofday(&stop, NULL);
  time(&end_t);
  diff_t = difftime(end_t, start_t);
  printf("Done executing GPU matmul\n");

  cudaMemcpy(bigOutput_host, bigOutput, bigN * sizeof(double), cudaMemcpyDeviceToHost);
  for(int i=0;i<bigN;i++) printf("%f ",bigOutput_host[i]);
  printf("\nTime: %f, %f\n", diff_t, end_t - start_t);
  printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec); 
}	
