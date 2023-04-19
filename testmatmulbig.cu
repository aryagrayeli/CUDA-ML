#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "matmul.cu"
#include "matmul.h"
#include <time.h>

int main() {

  srand(time(NULL));
  
  int power = 15;
  int bigN = (1<<power), bigM = (1<<14); 
  float *bigA, *bigB, *bigOutput;
  cudaMallocManaged(&bigA, bigN * bigM * sizeof(float));
  cudaMallocManaged(&bigB, bigM * sizeof(float));
  cudaMallocManaged(&bigOutput, bigN * sizeof(float));

  printf("successfully malloced a, b, output");

  for(int i=0;i<bigN*bigM;i++) bigA[i] = rand()/RAND_MAX;
  for(int i=0;i<bigM;i++) bigB[i] = rand()/RAND_MAx;
  
  printf("Test Big Matrix Multiplication\n");
  int threads = 1 << 6;
  dim3 bigGridSize((bigN + threads - 1)/threads, (bigM + threads - 1)/threads, 1);
  dim3 bigBlockSize(threads, threads, 1);
  matrix_mul<<<bigGridSize, bigBlockSize>>>(bigA, bigB, bigOutput, bigN, bigM);
  cudaDeviceSynchronize();
  printf("Done executing GPU matmul\n");

  for(int i=0;i<bigN;i++) printf("%f ",bigOutput[i]);
  printf("\n");
  
}	
