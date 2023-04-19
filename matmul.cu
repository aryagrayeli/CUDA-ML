#pragma once

#include <stdlib.h>
#include <math.h>

// Spawn N x M Threads
__global__ void matrix_mul(float ** a, float * b, float * output) {
    int n = gridDim.x;
    int m = blockDim.x;
    int r = blockIdx.x;
    int c = threadIdx.x;

    int Mi = 31 - __builtin_clz(m);
    int M = 1 << Mi;
    __shared__ float arr[M];
    __shared__ float B[m];
    cudaMemcpy(arr, a[r], sizeof(float) * m, cudaMemcpyDeviceToDevice);
    cudaMemcpy(B, b, sizeof(float) * m, cudaMemcpyDeviceToDevice);
	cudaMemset(arr + m, 0, sizeof(float) * (M - m));
    arr[c] *= B[c];
    __syncthreads();
    for (int i = 0; i < Mi; i++) {
        if (c >> (Mi - i - 1)) {
            continue;
        }
        int a = c << (i + 1);
        int b = a + (1 << i);
        arr[a] += arr[b]
        __syncthreads();
    }

    output[r] = *arr; 
}

// Spawn 1 x N Threads
__global__ void vector_sigmoid(float * input, float * output) {
    int i = threadIdx.x;
    output[i] = 1.0 / (1 + exp(-input[i]));
}

// Spawn 1 x N Threads
__global__ void vector_dsigmoid(float * input, float * output) {
    int i = threadIdx.x;
    float sigmoid = 1.0 / (1 + exp(-input[i]));
    output[i] = sigmoid * (1-sigmoid);
}

// Spawn 1 x N Threads
__global__ void vector_relu(float * input, float * output) {
    int i = threadIdx.x;
    output[i] = input[i] > 0 ? input[i] : 0;
}

// Spawn 1 x N Threads
__global__ void vector_drelu(float * input, float * output) {
    int i = threadIdx.x;
    output[i] = input[i] > 0;
}

// Spawn 1 x N Threads
__global__ void vector_tanh(float * input, float * output) {
    int i = threadIdx.x;
    output[i] = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));
}

// Spawn 1 x N Threads
__global__ void vector_dtanh(float * input, float * output) {
    int i = threadIdx.x;
    float tanh = (exp(input[i]) - exp(-input[i])) / (exp(input[i]) + exp(-input[i]));
    output[i] = 1 - tanh*tanh;
}

float* vector_softmax(float* input, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  float sum = 0;
  for(int i = 0; i < n; i++)
    sum += exp(input[i]);
  for(int i = 0; i < n; i++)
    output[i] = exp(input[i])/sum;
  
  return output;
}

float* vector_dsoftmax(float* input, int j, int n) {
  float* output = (float*) malloc(sizeof(float) * n);

  float sum = 0;
  for(int i = 0; i < n; i++)
    sum += exp(input[i]);
  float sj = exp(input[j])/sum;
  for(int i = 0; i < n; i++)
    if(i == j)
      output[i] = sj * (1-sj);
    else
      output[i] = -sj*(exp(input[i])/sum);
  
  return output;
}

// Spawn 1 x N Threads
__global__ void vector_add(float * a, float * b, float * output) {
    int i = threadIdx.x;
    output[i] = a[i] + b[i];
}

// Spawn N x M Threads
__global__ void matrix_hadamard(float ** a, float ** b, float ** output) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    output[r][c] = a[r][c] * b[r][c];
}

// Spawn N x M Threads
__global__ void matrix_trans(float ** input, float ** output) {
    int r = threadIdx.x;
    int c = blockIdx.x;
    output[r][c] = input[c][r];
}

// Spawn N x M Threads
__global__ void matrix_scalar(float ** input, int sc, float ** output) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    output[r][c] = sc * input[r][c];
}