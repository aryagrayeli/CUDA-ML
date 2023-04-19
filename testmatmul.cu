#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "matmul.cu"

int main() {
    int n = 3, m = 3;
    float *a, *b, *c;

    cudaMallocManaged(&a, n * m * sizeof(float));
    cudaMallocManaged(&b, m * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));

    // [1 -2 3]
    // [4 5 6]
    // [-7 8 9]
    a[0] = 1; a[1] = -2; a[2] = 3; a[3] = 4; a[4] = 5; a[5] = 6; a[6] = -7; a[7] = 8; a[8] = 9;
    b[0] = -1; b[1] = 2; b[2] = 3;

    dim3 gridSize(1, 1, 1);
    dim3 blockSize(n, m, 1);
    matrix_mul<<<gridSize, blockSize>>>(a, b, c, n, m);
    cudaDeviceSynchronize();

    printf("Test Matrix Multiplication\n");
    printf("Output: %f, %f, %f; Should be: 4, 24, 50\n\n", c[0], c[1], c[2]);
}