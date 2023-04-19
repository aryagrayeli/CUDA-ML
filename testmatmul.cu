#pragma once

#include <stdlib.h>
#include <math.h>

int main() {
    int n = 3, m = 3;

    float** A = (float**) malloc(sizeof(float*) * n);
    for(int i = 0; i < n; i++) A[i] = (float*) malloc(sizeof(float) * m);

    // [1 -2 3]
    // [4 5 6]
    // [-7 8 9]

    A[0][0] = 1; A[0][1] = -2; A[0][2] = 3; A[1][0] = 4; A[1][1] = 5; A[1][2] = 6; A[2][0] = -7; A[2][1] = 8; A[2][2] = 9;

    float** A_cuda;
    cudaMalloc(&A_cuda, sizeof(float*) * n);
    for (int i = 0; i < n; i++) {
        cudaMalloc(&A_cuda[i], sizeof(float) * m);
        cudaMemcpy(A_cuda[i], A[i], sizeof(float) * m, cudaMemcpyHostToDevice);
    }

    float* B = (float*) malloc(sizeof(float) * n);

    // [-1 2 3]

    B[0] = -1; B[1] = 2; B[2] = 3;

    float* B_cuda;
    cudaMalloc(&B_cuda, sizeof(float) * n);
    cudaMemcpy(B_cuda, B, sizeof(float) * n, cudaMemcpyHostToDevice);

    float* output_cuda;
    cudaMalloc(&output_cuda, sizeof(float) * n);

    matrix_mul<<<n, m>>>(A_cuda, B_cuda, output_cuda);

    float* output;
    cudaMemcpy(output, output_cuda, sizeof(float) * n, cudaMemcpyDeviceToHost);

    printf("Test Matrix Multiplication\n");
    printf("Output: %f, %f, %f; Should be: 4, 24, 50\n\n", output[0], output[1], output[2]);
}